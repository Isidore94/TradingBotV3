"""The read path: manifest-resolved canned queries (plan Phase 7, sec 8.3, 17).

Every supported read resolves its file list from ``manifest_log.jsonl`` at query
start and hands that explicit list to the reader. That is what makes a query
running across a compaction return either the pre- or the post-compaction row
set and never a double count: files are immutable, the COMPACT line is the
atomic switch, and superseded files stay readable for the 30-day ``_retired``
window. Globbing the lake directories is not a supported read path and is not
offered here.

What this module deliberately does NOT do (sec 17): no shrinkage, no intervals,
no evidence tiers, no rankings. The Phase-7 readout is raw canned-query
results - counts, mean R, and checkpoint values for the two slice setups - and
it is labelled EXPLORATORY so a thin cell is never read as a finding. The
Section 16.3 output contract arrives with milestone M-E.

DuckDB is optional and read-only. pyarrow answers every slice query; if duckdb
is installed it can execute SQL over the same manifest-resolved file list, and
if it is not, nothing degrades. Any ``.duckdb`` file would be a disposable
machine-local cache - never shared, never authoritative (LD-04).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

try:  # package import
    from .manifest import utc_now
    from .occurrences import SLICE_SETUPS, latest_occurrences
    from .outcomes import OUTCOME_DEFINITION_ID, is_matured
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    from manifest import utc_now  # type: ignore
    from occurrences import SLICE_SETUPS, latest_occurrences  # type: ignore
    from outcomes import OUTCOME_DEFINITION_ID, is_matured  # type: ignore
    from store import ResearchStore  # type: ignore

#: Every Phase-7 result carries this tier. Raw counts are not evidence.
EVIDENCE_TIER = "EXPLORATORY"
#: Coverage, latency, live-shadow, and promotion claims admit only these
#: capture modes (sec 9.3). The readout reports the split rather than
#: filtering silently, because the slice's D1 history is BACKFILL by nature.
AS_OBSERVED_MODES = ("LIVE", "DELAYED")

CHECKPOINT_COLUMNS = (
    "r_at_15m",
    "r_at_30m",
    "r_at_60m",
    "r_at_120m",
    "r_at_eod",
    "r_at_s1",
    "r_at_s2",
    "r_at_s3",
    "r_at_s5",
    "r_at_s10",
    "r_at_s18",
)


@dataclass
class QuerySnapshot:
    """A consistent view: rows plus the manifest position they came from."""

    manifest_seq: int = 0
    files: int = 0
    rows: list = field(default_factory=list)
    as_of: datetime | None = None
    evidence_tier: str = EVIDENCE_TIER


def read_snapshot(store: ResearchStore, dataset: str, partition: str | None = None) -> QuerySnapshot:
    """One dataset read, with the manifest position it was resolved at."""
    resolved = store.manifest.resolve(dataset=dataset, partition=partition)
    table = store.read_table(dataset, partition)
    return QuerySnapshot(
        manifest_seq=resolved.manifest_seq,
        files=len(resolved.entries),
        rows=table.to_pylist(),
        as_of=utc_now(),
    )


def dataset_inventory(store: ResearchStore | None) -> list[dict]:
    """Row and file counts per dataset, straight from the ledger."""
    if store is None:
        return []
    inventory: dict[str, dict] = {}
    for entry in store.manifest.resolve().entries:
        record = inventory.setdefault(entry.dataset, {"dataset": entry.dataset, "files": 0, "rows": 0})
        record["files"] += 1
        record["rows"] += entry.row_count
    return sorted(inventory.values(), key=lambda row: row["dataset"])


def _mean(values):
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def slice_readout(
    store: ResearchStore | None,
    *,
    year: int | None = None,
    as_of: datetime | None = None,
    outcome_definition_id: str = OUTCOME_DEFINITION_ID,
) -> QuerySnapshot:
    """The Phase-7 Research-tab table: raw results for the two slice setups.

    One row per (canonical_setup_id, side, recipe_id). Counts are reported
    three ways on purpose - rows, occurrences, and **episodes** - because only
    the last is a sample size, and matured outcomes are counted separately from
    open ones so an unresolved trade never flatters a mean.
    """
    snapshot = QuerySnapshot(as_of=as_of or utc_now())
    if store is None:
        return snapshot
    target_year = year or snapshot.as_of.year

    occurrences_by_id = latest_occurrences(store, target_year)
    resolved = store.manifest.resolve(dataset="outcome_path", partition=f"year={target_year}")
    snapshot.manifest_seq = resolved.manifest_seq
    snapshot.files = len(resolved.entries)
    outcome_rows = store.read_table("outcome_path", f"year={target_year}").to_pylist()

    grouped: dict[tuple[str, str, str], dict] = {}
    for outcome in outcome_rows:
        if str(outcome.get("outcome_definition_id")) != outcome_definition_id:
            continue
        occurrence = occurrences_by_id.get(str(outcome.get("occurrence_id")))
        if occurrence is None:
            continue
        setup = str(occurrence.get("canonical_setup_id") or "")
        if setup not in SLICE_SETUPS:
            continue
        key = (setup, str(occurrence.get("side") or ""), str(outcome.get("recipe_id") or ""))
        bucket = grouped.setdefault(
            key,
            {
                "canonical_setup_id": key[0],
                "side": key[1],
                "recipe_id": key[2],
                "outcome_definition_id": outcome_definition_id,
                "_outcomes": [],
                "_episodes": set(),
                "_symbols": set(),
                "_sessions": set(),
                "_modes": {},
            },
        )
        bucket["_outcomes"].append(outcome)
        bucket["_episodes"].add(occurrence.get("dependency_cluster_id"))
        bucket["_symbols"].add(occurrence.get("symbol"))
        entry_at = outcome.get("entry_at") or occurrence.get("event_at")
        if entry_at is not None:
            bucket["_sessions"].add(entry_at.date())
        mode = str(outcome.get("input_capture_mode_worst") or "")
        bucket["_modes"][mode] = bucket["_modes"].get(mode, 0) + 1

    rows = []
    for key in sorted(grouped):
        bucket = grouped[key]
        outcomes_list = bucket["_outcomes"]
        matured = [row for row in outcomes_list if is_matured(row, snapshot.as_of)]
        triggered = [row for row in matured if row.get("result_state") != "NO_TRIGGER"]
        row = {
            "canonical_setup_id": bucket["canonical_setup_id"],
            "side": bucket["side"],
            "recipe_id": bucket["recipe_id"],
            "outcome_definition_id": bucket["outcome_definition_id"],
            "n_rows": len(outcomes_list),
            "n_occurrences": len({str(row.get("occurrence_id")) for row in outcomes_list}),
            # The only one of these that is a sample size.
            "n_episodes": len(bucket["_episodes"] - {None}),
            "n_symbols": len(bucket["_symbols"] - {None}),
            "n_sessions": len(bucket["_sessions"]),
            "n_matured": len(matured),
            "n_open": len(outcomes_list) - len(matured),
            "n_no_trigger": len([row for row in outcomes_list if row.get("result_state") == "NO_TRIGGER"]),
            "mean_gross_r": _mean(row.get("gross_r") for row in triggered),
            "mean_net_r": _mean(row.get("net_r") for row in triggered),
            "mean_mfe_r": _mean(row.get("mfe_r") for row in triggered),
            "mean_mae_r": _mean(row.get("mae_r") for row in triggered),
            "capture_modes": dict(sorted(bucket["_modes"].items())),
            "as_observed_only": all(mode in AS_OBSERVED_MODES for mode in bucket["_modes"]),
            "evidence_tier": EVIDENCE_TIER,
        }
        for column in CHECKPOINT_COLUMNS:
            row[f"mean_{column}"] = _mean(item.get(column) for item in triggered)
        rows.append(row)
    snapshot.rows = rows
    return snapshot


def coverage_readout(store: ResearchStore | None, *, month: str | None = None) -> QuerySnapshot:
    """Scan coverage and gaps for one month, by status and reason."""
    snapshot = QuerySnapshot(as_of=utc_now())
    if store is None:
        return snapshot
    partition = month or f"month={snapshot.as_of:%Y-%m}"
    resolved = store.manifest.resolve(dataset="scan_coverage", partition=partition)
    snapshot.manifest_seq = resolved.manifest_seq
    snapshot.files = len(resolved.entries)

    statuses: dict[str, int] = {}
    risk_sets = set()
    for row in store.read_table("scan_coverage", partition).to_pylist():
        statuses[str(row.get("scan_status"))] = statuses.get(str(row.get("scan_status")), 0) + 1
        risk_sets.add(str(row.get("risk_set_id")))
    reasons: dict[str, int] = {}
    for row in store.read_table("collection_gap", partition).to_pylist():
        reasons[str(row.get("reason"))] = reasons.get(str(row.get("reason")), 0) + 1

    snapshot.rows = [
        {
            "partition": partition,
            "risk_sets": len(risk_sets - {""}),
            "coverage_by_status": dict(sorted(statuses.items())),
            "gaps_by_reason": dict(sorted(reasons.items())),
            "evidence_tier": EVIDENCE_TIER,
        }
    ]
    return snapshot


def render_slice_readout(snapshot: QuerySnapshot) -> str:
    """Plain-text rendering of the readout (CLI and Health surfaces)."""
    if not snapshot.rows:
        return "No slice outcomes recorded yet."
    header = f"{'setup':28} {'side':5} {'recipe':22} {'ep':>4} {'mat':>4} {'net R':>8} {'s18':>8}"
    lines = [header, "-" * len(header)]
    for row in snapshot.rows:
        lines.append(
            f"{row['canonical_setup_id'][:28]:28} {row['side'][:5]:5} {row['recipe_id'][:22]:22} "
            f"{row['n_episodes']:>4} {row['n_matured']:>4} "
            f"{_fmt(row['mean_net_r']):>8} {_fmt(row['mean_r_at_s18']):>8}"
        )
    lines.append("")
    lines.append(
        f"{snapshot.evidence_tier}: raw counts only - no shrinkage, no intervals, no ranking. "
        f"'ep' is independent episodes (the sample size), not rows."
    )
    return "\n".join(lines)


def _fmt(value) -> str:
    return "-" if value is None else f"{float(value):+.2f}"


# ---------------------------------------------------------------------------
# Optional DuckDB: read-only SQL over the same manifest-resolved file list
# ---------------------------------------------------------------------------
def duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
    except Exception:
        return False
    return True


def query_sql(store: ResearchStore, dataset: str, sql: str, *, partition: str | None = None):
    """Run read-only SQL over one dataset's manifest-resolved files.

    ``sql`` must reference the table name ``t``. Requires duckdb; callers check
    :func:`duckdb_available` first, because pyarrow answers every slice query
    and duckdb is a convenience, never a requirement (LD-04).
    """
    import duckdb

    paths = [str(path) for path in store.resolve_files(dataset, partition)]
    connection = duckdb.connect(database=":memory:")  # disposable, never shared
    try:
        if not paths:
            return []
        # DuckDB cannot prepare a DDL parameter, so the file list is inlined
        # after quoting. The list itself always comes from the manifest, never
        # from caller input or a directory glob.
        quoted = ", ".join("'" + path.replace("'", "''") + "'" for path in paths)
        connection.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet([{quoted}])")
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


__all__ = [
    "AS_OBSERVED_MODES",
    "CHECKPOINT_COLUMNS",
    "EVIDENCE_TIER",
    "QuerySnapshot",
    "coverage_readout",
    "dataset_inventory",
    "duckdb_available",
    "query_sql",
    "read_snapshot",
    "render_slice_readout",
    "slice_readout",
]
