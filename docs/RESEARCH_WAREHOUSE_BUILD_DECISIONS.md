# Research warehouse — builder decision log

Running log of the **implementation** decisions taken while building
[`docs/ULTIMATE_SETUP_DATABASE_PLAN.md`](ULTIMATE_SETUP_DATABASE_PLAN.md)
Phases 0-8. It exists so a reviewer (Sol, Fable, or the next builder) can audit
every choice that the locked plan did not already make, without reading the
diffs.

**Authority.** This file is subordinate to everything above it: root
[`plan.md`](../plan.md) → the locked warehouse plan (its Section 23 LD-01..LD-28
decision log and Section 7.1 frozen schemas) → `docs/decisions/` records → this
log. Nothing here may re-litigate an LD decision or a plan.md Section 5
invariant; where the plan is explicit, the plan wins and no BD entry exists.

**Scope.** A BD entry is written only where the plan left a genuine
implementation gap, where two readings of the plan were both defensible, or
where a choice binds future phases. Format: decision → why → what was rejected
→ reopen trigger → where it lives.

Status of the build: Phase 0-2 landed; Phase 3 in progress. Test baseline and
branch live in [`SOL_PROGRESS.md`](../SOL_PROGRESS.md).

---

## BD-01 — `schemas.py` landed in Phase 1, not later

**Decision.** The frozen Section 7.1 schemas (13 tables), the partition spec as
a dataset registry, and the deterministic identity hashes were written in
Phase 1, even though Section 19.2 lists no phase for `schemas.py`.

**Why.** The store cannot place a file or summarize `min_ts`/`max_ts` without
knowing a dataset's layer, partition dimensions, and time column. Section 7.1 is
frozen, so writing it is transcription, not design.

**Rejected.** A schema-free store taking raw pyarrow tables — it would have
pushed partitioning decisions out to every caller, which is exactly how the
locked partition spec drifts.

**Reopens if.** Never for the 13 tables; the forward-declared datasets of
Section 7.2 are added by their owning phase.

**Where.** `scripts/research_warehouse/schemas.py`;
`tests/test_warehouse_seal.py::test_dataset_registry_is_the_frozen_first_increment`.

## BD-02 — Layer mapping: seven information layers onto three directories

**Decision.** Section 4's layers fold onto the Section 8.2 directory contract
as: raw wraps → `bronze/`; normalized market facts (sessions, bars, universe,
anchors, levels, coverage, gaps) → `silver/`; feature/setup/style/gold layers →
`gold/`.

**Why.** Section 8.2 fixes exactly three data directories while Section 4 names
seven layers. Collapsing the upper four into `gold/` keeps the directory
contract literal and keeps the *dataset* the unit of identity, which is what the
manifest and partition spec are built on.

**Rejected.** Adding `feature/`, `setup/`, `style/` directories — that edits the
locked directory contract for cosmetic tidiness.

**Reopens if.** The plan's directory contract is amended.

**Where.** `schemas.py` `DatasetSpec.layer`;
`docs/RESEARCH_WAREHOUSE_ERD.md`.

## BD-03 — Crash between rename and manifest append: adopt, don't discard

**Decision.** Startup reconciliation adopts an orphan file (present in the live
tree, absent from the ledger) by appending a `PUBLISH` line marked
`reconciled: true` — unless its content hash is already registered live in that
partition, or the file is unreadable, in which case it is moved to
`_quarantine/`.

**Why.** The plan requires the crash to be "reconciled at startup" without
saying how. The file passed hash+validation before `os.replace`, so it is
complete evidence; deleting it would violate "never destroys the last verified
artifact", while blind adoption after a successful retry would double-count.
The content-hash check makes both failure modes impossible.

**Rejected.** (a) Quarantine every orphan — loses good evidence when the spool
segment has already been consumed. (b) A pre-intent record before the rename —
that is a fifth step in a 4-step protocol the plan fixes exactly.

**Reopens if.** A pilot session shows adopted-orphan rows arriving with content
that a retry had already published under a *different* byte layout (i.e. the
hash check stops being sufficient).

**Where.** `store.py::ResearchStore.reconcile`;
`tests/test_warehouse_seal.py` (adopt / duplicate / unreadable cases).

## BD-04 — Manifest corruption vs a torn final line

**Decision.** A malformed **final** line with no trailing newline is treated as
a crash artifact: ignored on read, truncated before the next append. A malformed
line anywhere earlier raises `ManifestCorruptionError`, which vetoes publishes
wholesale.

**Why.** The plan names manifest corruption as the *only* wholesale-veto
condition. A half-written last line is the ordinary consequence of losing power
mid-append and does not mean the ledger's history is untrustworthy; treating it
as corruption would black out capture for exactly the reason the tracker
incident is a pinned regression.

**Rejected.** Refusing to write until a human repairs any malformed line —
converts a routine crash into an outage.

**Reopens if.** A torn tail is ever observed carrying a *complete* JSON object
(would mean the append is not the atomic unit assumed here).

**Where.** `manifest.py::ManifestLog.read_entries` / `repair_torn_tail`;
`tests/test_warehouse_seal.py`, `tests/test_warehouse_manifest.py`.

## BD-05 — Quarantine payloads are JSONL, not Parquet

**Decision.** Quarantined records are written as JSON lines under
`_quarantine/<dataset>/<partition>/<symbol>/`, each with its reason and the
original row, and registered with a `QUARANTINE` manifest line.

**Why.** A row is quarantined precisely because it does not fit the typed
schema (bad type, naive timestamp, unresolvable partition key). Writing it as
Parquet under that schema is impossible; JSONL preserves it verbatim, which is
what "never silently discarded" requires.

**Rejected.** A generic quarantine Parquet table with a stringified payload —
more machinery, no more fidelity, and it invites treating quarantine as a
dataset instead of an incident.

**Reopens if.** Quarantine volume ever becomes large enough to need columnar
reads (it is an incident channel; that would itself be the signal).

**Where.** `store.py::_quarantine_rows`; `tests/test_warehouse_quarantine.py`.

## BD-06 — Naive timestamps are quarantined, never localized

**Decision.** A datetime without a timezone is a dirty row, not something to
interpret as UTC or exchange-local.

**Why.** plan.md Section 5 and the plan's Section 2: every timestamp is
timezone-aware and tied to an exchange session; missing information is
uncertainty, never confirmation. Guessing a zone would silently shift bars
across session boundaries.

**Rejected.** Assuming UTC (or exchange-local) for naive inputs.

**Reopens if.** Never.

**Where.** `store.py::_coerce_value`; `tests/test_warehouse_quarantine.py`.

## BD-07 — Bronze wraps live in a `bronze_*` dataset namespace

**Decision.** Each wrapped legacy artifact gets its own dataset
(`bronze_<artifact>`) sharing one record schema, partitioned by month, and
never a compaction input.

**Why.** Section 7.1 freezes the 13 **canonical** tables; Section 19.5 defines
bronze wrapping separately and does not enumerate its datasets. Giving each
artifact a dataset keeps the locked "one file per (dataset, month)" rule literal
and keeps one artifact's dirty tail from touching another's partition.

**Rejected.** (a) One giant `bronze_records` table partitioned by artifact —
invents a partition dimension the locked spec does not have. (b) Copying legacy
files byte-for-byte into the lake — the lake's write path is pyarrow-only,
forever (LD-04).

**Reopens if.** The plan enumerates bronze datasets explicitly.

**Where.** `schemas.py::bronze_dataset_spec`, `ingest_existing.py`;
`tests/test_warehouse_import.py`.

## BD-08 — The ingest watermark lives on the manifest line

**Decision.** Bronze idempotency state (`bronze_source_path`,
`bronze_source_sha256`, `bronze_max_offset`) is carried as extras on the seal's
own manifest line. There is no ingest-state file anywhere.

**Why.** The manifest is already the read authority and is append-only; a
side-car state file would be mutable state in an immutable lake and could drift
from what was actually sealed.

**Rejected.** A `definitions/ingest_state.json`, or deriving the watermark by
scanning the bronze Parquet each night (correct but needlessly expensive on the
~676 MB tracker and ~108 MB integrity log).

**Reopens if.** A source file is ever rewritten in place *and* its records must
still dedupe by content (today: append-only logs use the offset watermark,
rewritten documents use `SNAPSHOT` mode).

**Where.** `ingest_existing.py::_source_extras` / `_watermark`;
`tests/test_warehouse_import.py`.

## BD-09 — Snapshot artifacts compare against the last version, not all versions

**Decision.** A whole-document artifact (`price_alerts.json`, the watch JSONs,
heartbeat, tracker) ingests a new row whenever its content hash differs from the
**previous** ingest, including when it reverts to earlier content.

**Why.** These are trader-owned documents. "The trader put that level back" is a
real event in the history of their geometry; deduping against all past versions
would erase it.

**Rejected.** Deduping against every hash ever seen (chosen first, then
corrected before commit).

**Reopens if.** A high-churn artifact makes version rows dominate bronze volume.

**Where.** `ingest_existing.py::ingest_artifact`; `tests/test_warehouse_import.py`.

## BD-10 — Wrapped D1 bars record `provider = UNKNOWN`

**Decision.** Rows projected from the durable per-symbol D1 Parquet store carry
`provider = "UNKNOWN"` and `capture_mode = BACKFILL`.

**Why.** That store never persisted which provider produced a row (the source
is a frame attribute, not a column). The plan requires every row to record its
actual source and forbids silent blending; the honest value for unrecorded
provenance is UNKNOWN. Phase 3's tee records the real provider going forward.

**Rejected.** Defaulting to IBKR because it is primary — that is exactly the
silent provider blend risk R7 exists to prevent.

**Reopens if.** A provider column is added to the durable store, or a
sentinel-parity job can attribute historical rows with evidence.

**Where.** `ingest_existing.py::ingest_daily_bars`; `tests/test_warehouse_import.py`.

## BD-11 — Wrapped D1 ingest skips the current session

**Decision.** Only sessions strictly before the ingest date are projected into
`bar_d1`; today's row is picked up on a later run.

**Why.** The durable store can hold the current session's forming bar. Completed
bars only (plan.md Section 5, decision 0007) — a forming bar is preview, never
evidence. Skipping is safe because the run is idempotent and tomorrow's run
takes it.

**Rejected.** Marking today's row `is_complete=False` and storing it — the
Phase-2 dataset has no preview consumer, and storing preview rows in a
completed-bar dataset invites a later join that forgets to filter.

**Reopens if.** A preview consumer is registered (it would be a separate
feature ID per Section 5.4, not a change to this rule).

**Where.** `ingest_existing.py::ingest_daily_bars`; `tests/test_warehouse_import.py`.

## BD-12 — `exploration_cohort.txt` ships empty

**Decision.** The file is committed with its rules in a header comment and **no
symbols**, and every consumer treats empty as a clean no-op.

**Why.** The exploration cohort is load-bearing against champion-conditioning
bias — it is part of the research denominator. An agent-invented list would
silently define what the study population is. Item 5 of the plan's confirmation
register asks Aaron to supply it, and the register is explicitly non-blocking,
so an empty file is the correct interim state.

**Rejected.** Seeding it with liquid names "for now" — the placeholder would
outlive the intent and end up cited as evidence.

**Reopens if.** Aaron confirms the 30 symbols (fill the file; no code change).

**Where.** `scripts/research_warehouse/exploration_cohort.txt`;
`tests/test_warehouse_import.py`.

## BD-13 — Champion modules are reached through their own loaders

**Decision.** Reuse-as-is sources are read by calling the champion's own loader
(`master_avwap_lib.levels.load_level_store`, `d1_level_feed`'s AI-state
loader), with `scripts/` put on `sys.path` the same way `launch_gui.py` and
`smoke_check.py` do. The only thing mirrored rather than imported is the
symbol→filename sanitizer for the durable D1 store, because importing the legacy
scanner core would pull the GUI stack into a headless build job.

**Why.** "Reimplement nothing" (Section 19.0). Parsing those stores again would
create a second definition of the same geometry.

**Rejected.** Re-parsing the JSON directly; importing `master_avwap_lib.legacy`
for one private path helper.

**Reopens if.** The champion publishes a public path helper (switch to it), or a
loader signature changes (the wrapped read fails soft and the source is skipped,
never guessed).

**Where.** `ingest_existing.py::_hv_store_levels` / `_d1_feed_levels` /
`durable_daily_bar_file`.

## BD-14 — Test module names follow the plan's pytest checklist

**Decision.** New tests use exactly the Section 19.3 checklist module names
(`test_warehouse_seal.py`, `_manifest`, `_quarantine`, `_retire`, `_import`,
…). Coverage that the checklist does not name (e.g. the dataset registry) is
added to the nearest checklist module rather than to a new file.

**Why.** The checklist is the plan's own map of what must be proven; keeping
module names aligned means a reviewer can tick it off directly.

**Reopens if.** A phase needs a genuinely new test surface (add the module and
note it here).

**Where.** `tests/test_warehouse_*.py`.

## BD-15 — The tee reads BounceBot's in-memory cache; it is not a hook

**Decision.** The M5 tee takes the champion's existing
`latest_bars["<SYM>|5 D|5 mins"]` mapping as an argument and archives what is
already in it. `bar_archive.py` contains no provider client, no connection, no
retry, and no call site inside any champion fetch path.

**Why.** R3 requires the tee to observe in-memory responses only, and to add
zero requests. Reading a dict the champion already populated cannot change
champion timing, cannot fail a fetch, and cannot trip the Yahoo circuit
breaker, because it never makes a request that could fail.

**Rejected.** A callback the champion invokes after each fetch — even a
non-blocking one is capture code inside the fetch path, which R3 names as the
risk. A subclass or monkeypatch of the fetch method — worse, and invisible.

**Reopens if.** A cohort is needed that the champions never fetch — that is
what the Phase-3b nightly BACKFILL budget is for, not the tee.

**Where.** `bar_archive.py::extract_tee_bars` / `capture_m5_tee`;
`tests/test_warehouse_tee.py::test_tee_has_no_provider_client_at_all` (AST
check: no provider import, no request call anywhere in the module).

## BD-16 — Session identity in Phase 3 comes from `market_session`

**Decision.** `bar_m5` rows get their `session_id` (`XNYS-<date>`), RTH
boundaries, and PRE/RTH/POST phase from a wrapped read of the champion's
`scripts/market_session.py`. Phase 4 creates the matching `trading_session`
rows and owns the exchange-calendar version.

**Why.** Bars cannot be archived without a session identity, and inventing a
second session calendar beside the champion's would guarantee they disagree.
The `XNYS-<date>` shape is the plan's own documented `session_id` format.

**Reopens if.** Phase 4's calendar source differs from `market_session` (then
`trading_session` is authoritative and this wrapped read follows it).

**Where.** `bar_archive.py::session_context`; `tests/test_warehouse_tee.py`.

## BD-17 — IB round-lot volume is stored as provided

**Decision.** `bar_m5.volume` records exactly what the provider returned. IB
historical TRADES volume is in round lots while Yahoo is in shares; the
difference is disambiguated by the `provider` column and checked by the
sentinel-parity job, never normalized at capture.

**Why.** The plan says so for D1 ("the ×100 round-lot bug is a sentinel check,
not a rewrite") and the same reasoning applies to M5: rewriting captured
evidence to match an assumption is how the 2026-07-20 RVOL bug became
invisible. The champion's own ×100 conversion stays where it is, in the
champion.

**Reopens if.** Never for stored evidence; a derived shares-normalized column
would be a new, named feature.

**Where.** `bar_archive.py::capture_m5_tee`;
`tests/test_warehouse_tee.py::test_captured_rows_carry_session_phase_and_provenance`.

## BD-18 — `spool.py` landed with Phase 3, and a segment sheds whole

**Decision.** The spool (writer/sealer split, 5 GB / 7-day cap, fixed shedding
order) was built in Phase 3 rather than left unassigned, because live capture
needs a write target that is not the lake. Shedding operates on whole segments,
and a segment sheds only when **every** record in it carries that shed class; a
mixed segment is treated as PROTECTED.

**Why.** Section 8.4 states the ownership contract (GUI writer spools, CLI
build job seals) but Section 19.2 assigns `spool.py` no phase. Per-record
shedding inside a segment would mean rewriting an append-only file; whole-
segment shedding keeps the writer append-only, and the conservative treatment
of mixed segments means protected evidence can never be dropped as collateral.

**Rejected.** Writing the lake directly from the GUI session (breaks the stated
ownership split); per-record shedding (rewrites append-only files).

**Reopens if.** A measured DAS-outage day shows whole-segment granularity
sheds materially more than needed (LD-12's reopen trigger already covers the
cap itself).

**Where.** `scripts/research_warehouse/spool.py`;
`tests/test_warehouse_spool.py`.

## BD-19 — Shed evidence becomes a `collection_gap` row

**Decision.** Dropping a spool segment writes a shed-log record; the next seal
converts those into `collection_gap` rows with
`reason = NOT_COLLECTED_BY_POLICY`, `resolution = POLICY`.

**Why.** Section 8.4 requires gaps to be recorded and evidence never silently
deleted. Data the spool dropped under its declared cap policy is policy
absence — which is exactly the state the plan insists must never be conflated
with `MISSING`.

**Reopens if.** A shed class is ever introduced that is not policy-driven (it
would need its own reason code).

**Where.** `spool.py::_shed_segment` / `_seal_shed_log`;
`tests/test_warehouse_spool.py::test_shed_evidence_becomes_an_explicit_gap_row`.

## BD-20 — OPEN: the tee has no live caller yet

**Not a decision — a declared gap, so it cannot be mistaken for done.**

Phase 3 delivers the capture mechanism (tee → spool → seal) and its tests, but
nothing in the running desk calls it yet: `scripts/ui/services/warehouse_service.py`
(the GUI-owned service that would hand BounceBot's `latest_bars` to the tee each
production cycle, register the build job in the job ledger, and feed the six
Health tiles) is listed in the plan's module map without a phase, and its Health
tiles are Phase 8.

Consequence: until that service exists, capture runs only when a build job is
invoked manually. The 20-session pilot (Phase 8) depends on it, so it must land
no later than Phase 8 — earlier if the trader wants forward capture to start
sooner. Recorded here rather than silently deferred.

---

## Standing constraints this build re-checks every phase

Not decisions — the invariants each phase is audited against, listed so a
reviewer can spot-check them quickly:

1. Shadow-only: no detector, score, ranking, alert, or champion-timing path
   imports the warehouse package.
2. Total no-op when `research_store_dir` is unset (`ResearchStore.open()` →
   `None`, every entry point returns early).
3. The lake never lives inside the Drive home folder (config refuses it).
4. Completed bars only; missing data is uncertainty, never confirmation.
5. One owner per job; a failed publish never destroys the last verified artifact.
6. `calc_anchored_vwap_bands` is wrapped and parity-tested, never reimplemented
   (Phase 5).
7. The shared IB pacer never delays or queues champion traffic, and capture
   errors never count against the champion Yahoo circuit breaker (Phase 3/3b).
