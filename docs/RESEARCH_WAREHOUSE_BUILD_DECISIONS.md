# Research warehouse — builder decision log

Document role: **active warehouse decision log**. It records implementation choices;
the root `CHANGELOG.md` and `plan.md` own current status and priority.

Running log of the **implementation** decisions taken while building
[`ULTIMATE_SETUP_DATABASE_PLAN.md`](ULTIMATE_SETUP_DATABASE_PLAN.md)
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

Status of the build: Phases 0-8 landed (code); the 20-session pilot is a live
run that has not happened. Test baseline and branch live in
[`CURRENT_CHECKPOINT.md`](../CURRENT_CHECKPOINT.md). The 2026-08-04 review
([`RESEARCH_WAREHOUSE_REVIEW_2026-08-04.md`](RESEARCH_WAREHOUSE_REVIEW_2026-08-04.md)) repaired the outcome engine
(BD-53..BD-57) plus four mechanical defects (Windows lock probe, protected
spool shedding, spool re-seal dedup, capture reconnect). The follow-up defect
pass closed every remaining defect in that review — feature windowing (BD-58,
BD-59), backfill (BD-60), build-job coverage (BD-61), and the D14-D18 edge
cases (BD-62) — and wired the live tee (BD-63). The Windows Python 3.12 gate
and optional DuckDB install are now verified on the desk. What is left before
the pilot is not defect work: one broker-marked IB run (BD-25), live observation
of the tee/Health path, and the trader's confirmation-register items.

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

## BD-21 — The pacer is its own module and never opens a socket

**Decision.** The arbiter lives in `pacer.py` and only *decides*; the caller
acts. The module map lists "IB pacer integration" under `bar_archive.py`, but
`bar_archive.py` is the provider-free tee module (BD-15) and must stay that
way, so Phase 3b split three ways: `pacer.py` (arbiter), `backfill.py` (job
logic, provider-agnostic), `ib_capture.py` (the only socket).

**Why.** A decision-only arbiter is fully testable offline — the champion
pass-through property, the token bucket, the 162/366 yield, and the client-ID
allocation are all proven without a broker. Folding it into the tee module
would destroy the AST proof that the tee has no provider client.

**Rejected.** One `bar_archive.py` holding tee + pacer + backfill + seed: it
would put a provider client in the module whose whole guarantee is not having
one.

**Reopens if.** Never for the split; the responsibilities named in the module
map are all present, just in three files instead of one.

**Where.** `pacer.py`, `backfill.py`, `ib_capture.py`;
`tests/test_warehouse_pacer.py`, `tests/test_warehouse_backfill.py`.

## BD-22 — Capture isolation from the champion breaker is structural

**Decision.** Capture errors are tagged `capture=True` at the pacer and handled
there. Nothing in the warehouse imports, reads, or writes
`_IBKR_HISTORICAL_FAILURE_COUNT`, and capture never calls a champion fetch
function — so the champion's Yahoo-only circuit breaker cannot be tripped by
capture, by construction rather than by discipline.

**Why.** Risk R1 is the BF.B/LC blackout precedent: a silent downgrade of live
scans to Yahoo. A structural guarantee survives refactors that a "remember not
to" comment does not.

**Rejected.** Adding a capture-aware branch to the champion's
`_record_ibkr_historical_result` — that edits champion code to protect against
a coupling that simply should not exist.

**Reopens if.** Capture is ever routed through a champion fetch path (it must
not be).

**Where.** `pacer.py::note_error`;
`tests/test_warehouse_pacer.py::test_capture_errors_never_touch_the_champion_circuit_breaker`
(imports the champion module and asserts its counter is untouched after 15
capture pacing errors).

## BD-23 — A champion pacing error also backs capture off

**Decision.** IB error 162/366 puts capture into cool-off whether the error was
observed on capture traffic or on champion traffic. The champion is never
slowed by it.

**Why.** The pacing window is shared by the whole installation. A champion
hitting 162 means the window is already under pressure; continuing to spend
capture requests into it would make the champion's situation worse, which is
the one thing capture must never do.

**Reopens if.** Pilot measurement shows champion 162s that carry no capture
implication (they would need to stay counted but stop triggering cool-off).

**Where.** `pacer.py::note_error`;
`tests/test_warehouse_pacer.py::test_a_champion_pacing_error_also_backs_capture_off_but_not_the_champion`.

## BD-24 — Capture requests use `formatDate=2` (epoch UTC)

**Decision.** The capture connection asks IB for epoch-second timestamps; the
tee keeps reading the champion's `formatDate=1` naive local strings in the
champion's own convention.

**Why.** Capture is a separate connection with no legacy consumers, so there is
no reason to inherit a timestamp format that needs a timezone guess. An
unambiguous instant is exactly what the five-column PIT contract wants. The tee
cannot change format — it reads bars the champion already parsed.

**Rejected.** Matching `formatDate=1` for symmetry — symmetry with an ambiguity
is not a benefit.

**Reopens if.** A TWS version returns epochs the adapter cannot read (the
parser already falls back to the string formats).

**Where.** `ib_capture.py::historical_request` / `parse_bar`;
`tests/test_warehouse_backfill.py`.

## BD-25 — Backfill jobs take an injected fetcher

**Decision.** `run_backfill` / `run_nightly_backfill` /
`run_weekly_universe_sweep` / `run_yahoo_seed` receive a `fetcher` callable and
drive it. The IB and Yahoo adapters are separate and injectable.

**Why.** The parts that fail quietly — resume-without-duplicates across the TWS
restart, pacer gating, gap recording, seed ledger resume — are all logic, and
this makes every one of them provable offline with a fake. The socket layer
stays small enough to review.

**Caveat, stated plainly.** `ib_capture.build_ib_transport` (the real ibapi
client) is **not verified**: no offline test can exercise it, and no broker-
marked run has happened yet. Its live behaviour must be confirmed on the desk
before the pilot depends on it.

**Where.** `backfill.py`, `ib_capture.py`; `tests/test_warehouse_backfill.py`.

## BD-26 — The warehouse owns a versioned exchange calendar

**Decision.** `exchange_calendar.py` states NYSE sessions as **rules** (full
closures with the Saturday→Friday / Sunday→Monday observance, plus the three
13:00 ET early closes) under an explicit `calendar_version` recorded on every
`trading_session` row. Boundaries are computed in `America/New_York` and stored
in UTC.

**Why.** `trading_session` requires real sessions, and nothing in the repo
modelled holidays: `scripts/market_session.py` returns 09:30-16:00 for whatever
date it is given, Christmas included. That is correct for the live desk (it only
asks about today, while trading) and wrong for an archive, where "no bars on
Thanksgiving" must be distinguishable from missing data. Rules + a version means
a later correction bumps the version instead of silently re-dating history.

**Rejected.** (a) Adding `pandas_market_calendars` — a new dependency needs the
decision-0012 pinning path and an approval this phase does not have. (b) Teaching
`market_session.py` about holidays — that is champion code serving a different
purpose, and changing it would alter live behaviour for a research need.

**Verification.** Generated 2025-2027 and checked every date against the
published NYSE calendars, including the awkward ones: 2026 Independence Day
falls on a Saturday, so 3 July is the *observed closure* and there is no early
close; 2027 Christmas falls on a Saturday, so 24 December is the closure and
there is no Christmas Eve half day.

**Reopens if.** An unscheduled closure occurs (weather, a day of mourning) —
rules cannot know it, so it appears as a session with no bars, i.e. an honest
gap. Adding such a day means a dated override list and a version bump.

**Where.** `scripts/research_warehouse/exchange_calendar.py`;
`tests/test_warehouse_aggregate.py`.

## BD-27 — A bucket with no constituents is absent, not a zero bar

**Decision.** Aggregation emits a `bar_derived` row only when at least one M5
constituent exists. A short bucket is published with its real
`constituent_count` / `constituent_expected` and `quality = PARTIAL`; a bucket
with nothing in it produces no row.

**Why.** A synthesized zero-volume bar is a claim that the market was quiet.
Absence of data is uncertainty (plan.md sec 5), and the honest record of it is
the `collection_gap` row that Phase 3 already writes — not a fabricated bar
that later joins as if it were an observation.

**Reopens if.** A consumer needs a dense bar grid; it would build one from the
session calendar plus the gap rows, without inventing evidence in the store.

**Where.** `aggregate.py::derive_session_bars`;
`tests/test_warehouse_aggregate.py::test_missing_constituents_are_partial_never_averaged_away`.

## BD-28 — A complete stub is COMPLETE; the flag carries the difference

**Decision.** The 15:30-16:00 H1 stub with all six M5 constituents is
`is_complete=True`, `quality=COMPLETE`, `is_stub=True`,
`stub_duration_min=30`. Completeness answers "did every expected constituent
arrive"; the stub flag and its true duration answer "is this comparable with a
full hour".

**Why.** Conflating the two would either mark every session's last hourly bar
as defective (it is not — it is exactly what the exchange traded) or hide that
it is half the length of the others. The plan wants both facts, separately.

**Where.** `aggregate.py`;
`tests/test_warehouse_aggregate.py::test_the_h1_stub_keeps_its_true_duration`.

## BD-29 — A short week is flagged through `is_stub`, with no fake duration

**Decision.** W1 bars for holiday-shortened weeks set `is_stub=True` and leave
`stub_duration_min` null; the real signal is `constituent_expected` (sessions
the calendar says existed) beside `constituent_count`.

**Why.** `stub_duration_min` means minutes, and a "week duration in minutes" is
a fiction — a four-session week is not 1,560 minutes of anything meaningful.
The session counts say precisely what happened; inventing a duration would put
a number in the store that no consumer can safely use.

**Where.** `aggregate.py::build_weekly_bars`;
`tests/test_warehouse_aggregate.py::test_a_short_week_is_flagged`.

## BD-30 — Champion computations are called, not re-derived

**Decision.** `features.py` contains no indicator math. It calls
`calc_anchored_vwap_bands` (AVWAP + σ bands), `compute_indicator_frame` (D1
EMA 8/15/21 and SMA 50/100/200), and `BounceBot._calculate_vwap_bands` for the
intraday session VWAP ±1σ. The last one uses no instance state, so it is
called **unbound** rather than by standing up the BounceBot application, which
would drag the GUI and broker stack into a headless build job.

**Why.** The plan requires it for AVWAP ("reused as the computation,
parity-tested to 1e-9 — never reimplemented") and the same logic applies to
every other champion quantity: a second implementation that agrees today is a
trap tomorrow.

**Rejected.** Reimplementing EMA/SMA/VWAP "because they're standard" — the
champion's conventions (`adjust=False`, ohlc4 typical price, running-deviation
σ) are what its history is calibrated to.

**Where.** `features.py`; `tests/test_warehouse_avwap_parity.py` proves 1e-9
parity *and* asserts by AST that the wrapper contains no loop, no power
operator, and no `sqrt` anywhere in the module.

## BD-31 — `atr14` is the house method at the schema's declared length

**Decision.** `atr14` uses the champion's true-range definition and simple-mean
method (`compute_atr_from_ohlc`) over 14 bars.

**Why.** The frozen schema column is `atr14` while the scanner's own constant
is `ATR_LENGTH = 20`. Changing the column name would edit a frozen schema;
changing the method would introduce a second ATR convention. Same method, the
schema's length, stated here so the difference from the scanner's 20 is on the
record and not a surprise.

**Reopens if.** A registered study needs ATR(20) as a research feature — that
is an additive column, not a redefinition of this one.

**Where.** `features.py::atr`;
`tests/test_warehouse_features.py::test_atr14_matches_the_house_true_range_method`.

## BD-32 — Favorite-zone definitions are stated and versioned (CONFIRM-OR-AMEND)

**Decision.** The sec 6.2 block is frozen in the schema but the plan states only
what each column measures. v1 definitions, under `feature_set_version =
tier1_v1`:

| Column | v1 definition | Source |
|---|---|---|
| `favorite_zone_coord` | `(close − AVWAPE) / (UPPER_1 − AVWAPE)`, long-oriented | stated verbatim in the plan |
| `favorite_zone_residence_bars` | consecutive completed bars, ending at the snapshot, closing inside [AVWAPE, UPPER_1] | plan names the measure |
| `second_band_streak` | consecutive completed bars closing at/above UPPER_2 | plan names the measure |
| `first_dev_touch_order` | count of separate UPPER_1 touch episodes since the anchor (consecutive touching bars = one episode) | **builder's definition** |
| `band1_rejection_strength` | on the most recent touching bar, `(high − close) / (high − low)`, clipped to [0, 1] | **builder's definition** |

**Why.** The columns exist in a frozen schema and Phase 5's deliverable is
"exactly the frozen columns". Leaving two of them permanently null would ship a
schema that cannot be filled; inventing definitions silently would let a number
nobody agreed on become cited evidence. Stating them under a feature-set version
gives the trader something concrete to amend, and a later definition is a
version bump plus additive rows — history is never rewritten.

**Trader action.** `first_dev_touch_order` and `band1_rejection_strength` are
confirm-or-amend items. Everything else in the block follows the plan's own
wording.

**Where.** `features.py::favorite_zone_block`;
`tests/test_warehouse_features.py::test_favorite_zone_definitions`.

## BD-33 — Production context is stored verbatim, never recomputed

**Decision.** `rvol_tc2000`, `rvol_gate_pass`, `rs_rw_vs_spy`,
`group_rs_debiased`, `market_internals_negative`, `session_structure_gate`, and
`pullback_count_in_current_leg` are accepted as inputs from wrapped production
evidence and written through unchanged. With no evidence they stay null.

**Why.** Section 6.8 calls these six shipped systems tier-1 and says they are
*migrated, not reinvented*. Recomputing an RVOL here would create a second
number with the same name — precisely the failure mode behind the 2026-07-20
round-lot bug.

**Consequence, stated plainly.** Until Phase 6 wires the bounce-ledger join,
these columns are null in practice. That is an honest gap, not a silent one.

**Where.** `features.py::compute_intraday_features`;
`tests/test_warehouse_features.py::test_production_context_is_stored_verbatim_never_recomputed`.

## BD-34 — Session VWAP ships the STANDARD algorithm first

**Decision.** Intraday snapshots carry `vwap_algorithm = "STANDARD"` with ±1σ.
DYNAMIC and EOD arrive when their champion functions are wrapped.

**Why.** Section 6.3 freezes tier-1 at the three production algorithms with ±1σ
only. `_calculate_vwap_bands` covers the standard reset-based one cleanly; the
dynamic and EOD variants live behind more champion state. `vwap_algorithm` is a
column, so the other two are additive **rows** later — no schema change, no
rewrite.

**Where.** `features.py::session_vwap_bands`.

## BD-35 — An intraday row keyed at bar S describes state through S's close

**Decision.** `feature_snapshot_intraday` rows are keyed by a completed M5
`interval_start`; bars up to and including that bar contribute, nothing later.

**Why.** The dataset grain is `symbol × M5 interval_start`, and the row is
computed once that bar completes. The first implementation keyed rows by a
bar's `interval_end`, which silently pulled the *next* bar into the row — a
real point-in-time leak, caught by the test that recomputes from truncated
history and demands an identical row.

**Where.** `features.py::compute_intraday_features`;
`tests/test_warehouse_features.py::test_intraday_snapshot_never_sees_a_later_bar`.

## BD-36 — The AVWAP golden fixture carries the Milestone-3 contract

**Decision.** `tests/fixtures/warehouse_avwap_bands_v1.json` is a
contract-bearing fixture (raw-input hash re-verified on load, declared 1e-9
tolerance, provider assumptions, as-of). It characterizes the champion: a
mismatch means the champion formula changed and must be reviewed, never that
the fixture should be regenerated.

**Why.** The repo already enforces that every shipped fixture bears the
contract (`test_fixture_contract.py`), and the contract's own loader applies
the declared tolerance — better than a hand-rolled comparison.

**Where.** `tests/fixtures/warehouse_avwap_bands_v1.json`,
`tests/test_warehouse_avwap_parity.py`.

## BD-37 — "Rescan updates" means a new revision, not a mutated row

**Decision.** A rescan of a live thesis recomputes the same `occurrence_id`. If
nothing in the snapshot fields changed, **nothing is written at all**. If
something changed, a new row is appended with `revision_id = rev-N+1` and
`supersedes_revision_id` pointing at the previous one; readers take the highest
revision (`latest_occurrences`). `first_detected_run_id` and the original
`event_at` are carried forward.

**Why.** The lake is append-only and immutable — a row cannot be edited in
place — while sec 7.3 requires that a rescan "updates a snapshot and never
creates an extra occurrence". Revisions satisfy both: the row count grows with
knowledge, the *occurrence* and *episode* counts do not. `episode_counts()`
reports rows, occurrences, and episodes separately so nobody can quote the
first as a sample size.

**Rejected.** Overwriting the row (impossible in an immutable lake, and it
would destroy the knowledge trail); appending an unlinked row per scan (that is
exactly the episode inflation of risk R9).

**Where.** `occurrences.py::record_occurrences`;
`tests/test_warehouse_occurrence.py` — 100 rescans leave 1 row, 1 occurrence,
1 episode.

## BD-38 — `dependency_cluster_id` excludes the setup family, includes the side

**Decision.** The episode key is `hash(symbol, side, structural_timeframe,
anchor-or-episode-start)`. Setup family is deliberately not an input; side is.

**Why.** Sec 7.3 says simultaneous AVWAP/band/MA/level variants attached to one
underlying move are several hypotheses about **one** episode, so family must not
split them. A long thesis and a short thesis on the same symbol are not the same
move, so side must not merge them.

**Reopens if.** A registered study needs long/short pairs treated as one episode
(e.g. a hedged construct) — that would be a second, named cluster definition,
not a redefinition of this one.

**Where.** `occurrences.py::dependency_cluster_id`;
`tests/test_warehouse_occurrence.py::test_variants_of_one_move_share_a_dependency_cluster`.

## BD-39 — The signal-close entry is a declared precommitted MOC

**Decision.** `swing_house_v1` and the two controls fill at the signal bar's
close, recorded as `entry = signal_close_precommitted_moc`, with
`signal_known_at == entry_eligible_at`.

**Why.** Sec 19.3 defines the house recipe as signal-close entry, while sec 12.1
forbids *assuming* a same-close fill "unless the recipe explicitly models a
precommitted market-on-close order". The recipe therefore declares one, in its
own `entry` field, so the assumption is visible on every row rather than buried
in the simulator.

**Reopens if.** The trader wants a next-open variant — that is a new
`recipe_id`, not a change to this one.

**Where.** `outcomes.py::SWING_HOUSE_V1`.

## BD-40 — Ambiguous bars: primary = STOP_FIRST, optimistic read kept as a bound

**Decision.** When one session's range contains both the stop and the target,
the row is `result_state = AMBIGUOUS_BAR`, `path_resolution = AMBIGUOUS`,
`first_hit = STOP`, `r_lower_bound` = the stop reading, `r_upper_bound` = the
target reading, and `gross_r` equals the lower bound.

**Why.** Sec 14.2 names STOP_FIRST the primary and requires the TARGET_FIRST
reading be retained as `r_upper_bound`. Averaging the two, or silently picking
the better one, would put an unearned number in the evidence base.

**Where.** `outcomes.py::simulate_swing`;
`tests/test_warehouse_outcomes.py::test_same_bar_ambiguity_is_stop_first_with_the_target_bound_kept`.

## BD-41 — Maturity is a calendar fact; missing path is TRUNCATED

**Decision.** `maturity_at` is projected on the exchange calendar even when the
lake holds fewer bars than the time stop. If the clock says an outcome should
have resolved but the bars ran out, the state is `TRUNCATED`, never `EXPIRED`
and never a silent `OPEN`.

**Why.** Maturity answers "should this have resolved by now", which is a
property of the clock, not of how much data arrived. Reporting a data shortfall
as a finished trade is how an incomplete archive turns into a fake result.

**Where.** `outcomes.py::_swing_maturity`;
`tests/test_warehouse_outcomes.py::test_maturity_is_a_calendar_fact_not_a_data_artifact`.

## BD-42 — House management is simulated, not just declared

**Decision.** `partial_at_band2_trail_band1_run_band3` is executed: 50% exits at
band 2, the remainder runs to band 3 or exits on a close back through band 1,
and `gross_r` is the weighted result. Without band levels the recipe falls back
to its stop/target path rather than silently claiming the managed result.

**Why.** Declaring a management policy in a recipe name while simulating a
plain stop/target would make every reported R wrong in the same direction. The
plan states the policy concretely enough to implement, so it is implemented.

**Caveat.** Partial fills are modelled at the band price with no slippage beyond
the `house_default_v1` cost model; band levels come from the Phase-5 AVWAP
block and are absent when the occurrence has no anchor.

**Where.** `outcomes.py::_house_management_r`; two arithmetic tests
(run-to-band-3 and trail-to-band-1).

## BD-44 — BUILT: tracker transition/scenario adapter

**Decision.** `tracker_adapter.py` streams the scenario CSV for first-seen
geometry and reads the small append-only setup-tracker transition ledger for
current lifecycle state. It never opens or parses the 1 GB tracker snapshot.
All canonical families in `setup_tagging._FAMILY_TAGS` are admitted when their
scenario row is tradeable and has valid stop geometry. Daily rescans collapse
by symbol, side, canonical family and anchor date; first-seen trigger/geometry
is frozen, while the latest ledger state supplies status. Simultaneous family
variants share one symbol/side/anchor dependency cluster.

**Why.** The transition ledger is the point-in-time record and the scenario CSV
already owns the exact structural stop source that the tracker measured. The
snapshot is current state, too large for the in-process build, and unnecessary.
Freezing first geometry prevents a later daily rescan from leaking future data.

**Measured audit, 2026-08-27.** 249,438 scenario rows + 10,820 transition rows
produced 6,663 deduplicated detections across all 16 registered families, with
zero unknown-family skips. This is an adapter audit, not outcome evidence.

**Reopens if.** Either source loses stable setup/family identity or structural
geometry, or the tracker changes from daily-rescan lifecycle semantics.

**Where.** `tracker_adapter.py`; `occurrences.py`; pinned in
`tests/test_setup_research_pipeline.py`.

## BD-43 — `intraday_bounce_v1` requires a linked bounce event

**Decision.** No linked BounceBot M5 event ⇒ no `intraday_bounce_v1` row, and
the report records `NO_LINKED_BOUNCE_EVENT`. The link is symbol + session +
a ±60-minute window around the trigger; the event supplies the bounce bar and
the production stop.

**Why.** Sec 19.3 is normative here, and the deeper rule is that the warehouse
never re-detects: manufacturing a bounce to fill a row would invent evidence.

**Reopens if.** The bounce ledger gains an explicit occurrence link — then the
time-window join is replaced by the real key.

**Where.** `occurrences.py::link_bounce_events`,
`outcomes.py::build_outcomes`;
`tests/test_warehouse_outcomes.py::test_no_linked_bounce_event_means_no_intraday_row`.

## BD-45 — DuckDB is pinned but strictly optional

**Decision.** `duckdb==1.5.5` is added to `requirements-dev.txt` and
`constraints.txt`; the read path answers every slice query through
`pyarrow.dataset`, and `query_sql` is a convenience that callers reach only
after `duckdb_available()` returns True.

**Why.** LD-04 defers DuckDB to Phase 7 behind "cp314 win_amd64 wheel verified
first". That wheel exists (`duckdb-1.5.5-cp314-cp314-win_amd64.whl` on PyPI, checked
2026-08-04), so the precondition is met — but the plan also says the fallback is
to stay on pyarrow, so nothing may depend on duckdb being installed. Its
connection is `:memory:` and disposable: no `.duckdb` file is ever created,
shared, or treated as authoritative.

**Desk verification addendum (2026-08-09).** `duckdb==1.5.5` was installed in the
uv-managed Windows Python 3.12 desk environment and the full suite passed with the
DuckDB query tests active. It remains optional; the pyarrow-only contract is
unchanged.

**Where.** `queries.py::query_sql` / `duckdb_available`; `requirements-dev.txt`,
`constraints.txt`; `tests/test_warehouse_queries.py` (the duckdb test skips
cleanly when it is absent).

## BD-46 — The readout reports rows, occurrences, and episodes separately

**Decision.** Every readout row carries `n_rows`, `n_occurrences`, and
`n_episodes`, plus `n_matured` / `n_open` / `n_no_trigger`, and means are
computed **only** over matured, triggered outcomes. The capture-mode split is
shown beside each row with an `as_observed_only` flag.

**Why.** Three recipes on one move produce three rows and one episode; only the
episode count is a sample size. An unresolved trade must not flatter a mean, and
the slice's D1 history is BACKFILL by nature — so rather than filter silently,
the readout shows the split and lets the reader see that these are not
as-observed claims.

**Where.** `queries.py::slice_readout`; `tests/test_warehouse_queries.py`.

## BD-47 — The Research tab reads only on demand

**Decision.** `WarehouseReadoutPanel` performs no lake read when constructed;
Refresh is the only path that opens the store. A disabled or broken warehouse
becomes a status message, never an exception.

**Why.** Section 20: "No GUI render path performs provider or large warehouse
reads." A panel that queried on construction would put a multi-GB lake read on
the Trading Desk's startup path.

**Where.** `scripts/ui/panels/warehouse_readout_panel.py`;
`tests/test_qt_warehouse_readout.py`.

## BD-48 — Qt tests live in `test_qt_*.py` with a module-level application

**Decision.** The readout panel's tests moved out of
`test_warehouse_queries.py` into `tests/test_qt_warehouse_readout.py`, using the
repo's existing shape: `QApplication` created once at import.

**Why.** With the panel test creating its application mid-test inside a non-Qt
module, the full suite passed (2027 tests, 67s) but the pytest process then
never exited — non-daemon `multitasking` pool threads from yfinance were left
alive. Following the repo's Qt-test shape fixes it; the suite now exits cleanly.
Worth recording because the symptom looks like a hung test and is not one.

**Where.** `tests/test_qt_warehouse_readout.py`.

## BD-49 — Backups are portable copies, not robocopy invocations

**Decision.** `backup.py` implements the three classes with `shutil` copies
(size-compared, skip-if-unchanged) rather than shelling out to `robocopy`.

**Why.** The plan names robocopy because the desk is Windows, but the property
that matters is *append-only incremental*: never propagate a deletion,
never `/MIR`. A portable implementation states that property in code, keeps the
restore check runnable in tests, and works on the macOS path the repo also
supports. `deleted_from_target` is always 0 and is asserted.

**Reopens if.** Copy throughput on the real lake makes a native tool worth it —
then the same no-delete contract must hold, and the test stays.

**Where.** `backup.py`; `tests/test_warehouse_restore.py` (a file deleted at
the source stays in the backup).

## BD-50 — A live lock refuses even inside the same process

**Decision.** `single_flight` refuses whenever the recorded PID is alive,
including when that PID is the current process. A lock whose holder is dead is
reclaimed rather than obeyed.

**Why.** The first cut allowed re-entry for the same PID, which would let a
build nest inside a build and write the lake twice concurrently — the exact
thing the single-flight rule exists to prevent. Reclaiming a dead holder's lock
is the other half: a crashed build must not wedge every future run.

**Where.** `cli.py::single_flight`; `tests/test_warehouse_restore.py`.

## BD-51 — Policy absence is not a coverage defect

**Decision.** The coverage Health tile counts `PARTIAL`/`MISSING` shortfalls but
excludes `NOT_COLLECTED_BY_POLICY` rows from the defect count, reporting them in
the tile's reason breakdown instead.

**Why.** A symbol the capture policy never intended to collect intraday is a
declared decision, not a hole. Colouring the tile amber for it would train the
trader to ignore the tile — which is how a real gap gets missed.

**Where.** `scripts/ui/services/warehouse_service.py::_coverage_tile`;
`tests/test_warehouse_restore.py::test_policy_absence_is_not_a_coverage_defect`.

## BD-52 — OPEN: the pilot is a live run, and the tee still has no caller

**Not a decision — the honest state of Phase 8.**

The Phase-8 *code* is done: three-class backup, the scripted restore check, the
CLI build job with its single-flight lock and job-ledger registration, and the
six Health tiles. Two things it cannot deliver from here:

1. **The 20-session pilot** (sec 5.6) is 20 forward RTH sessions on the desk.
   Its checklist items — measured req/min without error 162, the real line cap,
   stream-vs-poll M5 parity, idempotent resume across the TWS restart, measured
   bytes/row, DAS/backup health, the restore check — need live sessions.
2. **BD-20 is still open.** `warehouse_service.py` now exists (tiles + job
   descriptor), but nothing calls `capture_m5_tee` during a live session: the
   GUI still needs to hand BounceBot's `latest_bars` to the tee each cycle, and
   the Health page still needs to render these tiles. Until that lands, capture
   only runs when the build job is invoked by hand — which means the pilot
   cannot start.

**Where.** `cli.py`, `backup.py`, `scripts/ui/services/warehouse_service.py`.

## BD-53 — Non-terminal outcomes are recomputed and superseded by `computed_at`

**Decision.** `outcome_path` rows in a terminal state (`STOPPED`, `TARGETED`,
`EXPIRED`, `AMBIGUOUS_BAR`, `CENSORED` — `TERMINAL_RESULT_STATES`) are final
evidence and are never recomputed. Non-terminal rows (`OPEN`, `TRUNCATED`,
`NO_TRIGGER`) are re-simulated on every build against the bars now available;
a changed result is appended as a superseding row and an unchanged one writes
nothing. Because the frozen sec 7.1 schema gives `outcome_path` no revision
columns, supersession is by time: every reader takes the latest `computed_at`
per (occurrence, recipe, outcome_definition) via `latest_outcomes()`, across
**all** year partitions (the dataset partitions on `computed_at`, so a
recomputed row can live in a later year than its predecessor). Additionally,
an unresolved row carries **no realized R**: `gross_r`/`net_r` are null on
`OPEN` and `TRUNCATED` rows — checkpoints and MFE/MAE remain as path facts.

**Why.** The first implementation skipped any existing (occurrence, recipe,
definition) key outright, freezing every outcome at its first simulation: a
row computed two sessions after trigger kept its interim R forever, and once
`maturity_at` passed, the readout counted that interim number as matured
evidence (review defect D1, reproduced at +1.0R interim vs −0.8R actual).
Sec 14.2's own maturity discipline ("an unresolved label cannot enter
training … until matured") is unimplementable if the stored label can never
change.

**Rejected.** Simulating only after maturity — it would blind the readout to
open positions and still mislabel truncated archives; mutating rows in place —
impossible in an immutable lake; adding revision columns — edits a frozen
schema when `computed_at` already orders knowledge.

**Reopens if.** A registered study needs the interim path readings themselves
(they would become explicit checkpoint columns, never `gross_r`).

**Where.** `outcomes.py::build_outcomes` / `latest_outcomes` /
`TERMINAL_RESULT_STATES`; `queries.py::slice_readout` (reads the latest view;
`n_truncated` reported; `n_open` counts pending trades, not `NO_TRIGGER`);
`tests/test_warehouse_outcomes.py::test_an_open_outcome_is_resimulated_and_superseded`,
`tests/test_warehouse_queries.py::test_a_recomputed_outcome_supersedes_its_interim_reading`.

## BD-54 — House management is one bounded walk, and a stop after the partial keeps the partial

**Decision.** `partial_at_band2_trail_band1_run_band3` is simulated as a
single walk over the recipe's own window (`forward[:18]`) — bars past the
time stop contribute nothing, not even a band touch. Within the walk:
intra-bar favourable fills (band-2 partial; band-3 runner exit, which implies
band 2 was crossed first) precede close events (the structural close-failure
stop; after the partial, the band-1 trail). A stop or trail that fires after
the partial exits only the remaining half at that close and the partial stays
credited — `gross_r = 0.5·partial + 0.5·exit`. Result states map: band-3
completion → `TARGETED`; trail or stop exit → `STOPPED`; time stop →
`EXPIRED`; `first_hit` records the first fill-or-exit event. Same-bar
conservatism keeps the LD-07/BD-40 doctrine where it genuinely applies: a bar
that offers a NEW favourable fill, ends the position on its close, **and
touched the stop level intra-bar** is `AMBIGUOUS_BAR` — the primary reading
takes the exit without that bar's fills and the fill-credited reading is kept
in `r_upper_bound`. A pure band-1 trail with no stop touch is *not*
ambiguous: an intra-bar band fill definitionally precedes the bar's close.

**Why.** The Phase-6 implementation ran management over all forward bars
(crediting a band-2 touch on session 25 of an 18-session recipe, review
defect D3) and skipped management entirely when the main loop said `STOPPED`
(reporting a full-size stop after a 50% partial at +2R — review defect D4,
reproduced at −1.4R against the policy's +0.45R). BD-42's claim required the
declared policy to actually be executed.

**Reopens if.** The trader amends the management description (a new
`recipe_id`, per BD-39's pattern), or sec 12.2's "band-bounce stop one band
beyond" is wired in (it still comes from the detector's declared geometry).

**Where.** `outcomes.py::_walk_managed` / `_walk_plain` / `simulate_swing`;
`tests/test_warehouse_outcomes.py::test_management_credits_the_partial_when_the_runner_stops_out`
/ `test_management_never_looks_past_the_time_stop`.

## BD-55 — Maturity is `min(resolution, time stop)`, for swing and intraday alike

**Decision.** `maturity_at` on a resolved row is the resolution time (the
session close that completed the stop/trail/target, or the time-stop close
for `EXPIRED`); the projected 18-session date is used only while the path is
unresolved (`OPEN`/`TRUNCATED`). Intraday: a stopped bounce matures at the
stop bar's `interval_end`, else at the session close.

**Why.** Sec 14.2 says maturity is `min(+18 sessions, stop/target/expiry)`;
the first implementation always projected the full time stop, so a trade that
stopped on session 2 was excluded from every matured mean for weeks (review
defect D11).

**Where.** `outcomes.py::simulate_swing` / `simulate_intraday_bounce`;
`tests/test_warehouse_outcomes.py::test_maturity_is_min_of_resolution_and_the_time_stop`.

## BD-56 — Entry slippage and gap-through stops are modelled, not waived

**Decision.** Sec 14.2's "+1 half_spread slippage on stop/market entries" is
implemented as `net_r`'s `entry_slippage_half_spreads` term
(`ENTRY_SLIPPAGE_HALF_SPREADS = 1.0`), paid by every slice recipe because
every slice entry is a declared market-type order (the precommitted MOC,
BD-39; the completed-bounce-bar close). A future limit-entry recipe passes 0.
Intraday, a bar that opens beyond the stop fills at its **open**, not at the
level the market never traded (sec 14.3 gap-through-stop behaviour); swing
stops are close-based, so their exit close already carries the gap.

**Why.** The cost formula alone omitted the contract's separate slippage
bullet, and the intraday stop always filled at exactly −1.0R — both
systematically optimistic under a contract that says every deviation is a new
`outcome_definition_id` (review defect D12).

**Reopens if.** The trader's fills show the one-half-spread assumption is
materially wrong in either direction (that is a measured revision to
`house_default_v1`'s successor, never a silent retune).

**Where.** `outcomes.py::net_r` / `simulate_intraday_bounce`;
`tests/test_warehouse_outcomes.py::test_net_r_includes_the_market_entry_slippage_bullet`
/ `test_intraday_gap_through_the_stop_fills_at_the_open`.

## BD-57 — The intraday walk is bounded to the entry session and knows it is unresolved

**Decision.** `simulate_intraday_bounce` filters its bars to the entry
session's RTH window before walking — a bar from any later session is not
part of an EOD recipe's outcome, and `r_at_eod` (≡ `entry_r`) means the entry
session's close, never the last bar the caller happened to pass
(`_fill_intraday_checkpoints` takes a `session_close` bound; under a
signal-close MOC swing entry the intraday checkpoints are therefore null,
which is the honest value). A simulation run mid-session reports `OPEN`; a
session that closed but whose archived bars stop early reports `TRUNCATED`;
both carry no realized R and are recomputed later under BD-53.

**Why.** The first implementation walked every provided bar across sessions,
set `r_at_eod` from the final provided bar, and labelled a live trade
`EXPIRED` at whatever bar came last when run intraday (review defect D13).

**Where.** `outcomes.py::simulate_intraday_bounce` /
`_fill_intraday_checkpoints`;
`tests/test_warehouse_outcomes.py::test_intraday_bounce_never_walks_past_the_entry_session`
/ `test_intraday_bounce_mid_session_is_open_and_a_short_archive_is_truncated`.

## BD-58 — The D1 feature window is history-deep, and `tier1_v1` is corrected in place

**Decision.** `build_daily_snapshots` no longer reads a calendar partition. The
stated `tier1_v1` window rule is: **always** read `bar_d1` for `year` and
`year-1`, then keep walking back one year at a time until the deepest symbol in
the frame holds `DAILY_HISTORY_MIN_SESSIONS = 250` completed sessions on or
before the snapshot's `session_date` — stopping early when the lake runs out of
years, and never reading more than `DAILY_HISTORY_MAX_YEARS = 5`. The partitions
actually read are returned alongside the rows so `input_manifest_hash` covers
exactly the files the snapshot was computed from. Because no capture has ever
run — the lake holds zero `feature_snapshot_daily` rows — this is a correction
to `tier1_v1` **in place**, not a version bump.

**Why.** The old rule added the prior year only in January, but a 200-session
window spans roughly 9.5 calendar months. From February to mid-October the
frame was truncated, and the champion `compute_indicator_frame` uses
`rolling(period)` with pandas' default `min_periods`: `sma200` was silently
null for most of the year and `sma100` until about May. Worse, that function's
EMAs use `adjust=False` and are seeded at the frame's *first* bar, so `ema8/15/
21` were different numbers from the champion's under the champion's own column
names — the exact BD-33 failure (review defect D5). 250 sessions covers
`sma200` with margin and drives the EMA-21 seed error below float tolerance
(`(1 - 2/22) ** 250` is about 5e-11); it also exceeds the champion's own
deepest D1 fetch (`PRIORITY_SMA_LOOKBACK_DAYS = 320` calendar days, about 220
sessions), so the warehouse never sees less history than the scanner does.

**Rejected.** Bumping to `tier1_v2` — a version bump exists to protect an
existing corpus, and there is none; it would imply a `tier1_v1` body of
evidence worth comparing against and would permanently carry a version whose
only content is a bug. Reading a fixed two years unconditionally — cheap in the
common case but wrong for a partially seeded lake, where the honest answer is
"walk until deep enough or the lake ends". Passing `min_periods=1` to the
champion's rolling call — that re-derives champion math with different
semantics, which plan.md sec 5 forbids.

**Reopens if.** A tier-2 feature needs a window deeper than 250 sessions (raise
the constant; it is a floor, not a definition), or the champion's own lookback
constants change.

**Where.** `features.py::daily_history_window` / `build_daily_snapshots` /
`DAILY_HISTORY_MIN_SESSIONS`;
`tests/test_warehouse_features.py::test_a_midyear_daily_snapshot_matches_the_champions_full_history_frame`
/ `test_the_daily_window_always_reads_year_and_the_prior_year`.

## BD-59 — The intraday EMA lookback is the session, because that is the champion's frame

**Status: CONFIRMED by the trader (2026-08-04).** This entry contradicted the
2026-08-04 review, so it was written as a builder disagreement pending
adjudication. Aaron read the evidence and confirmed the champion reading and
the session-scoped decision. It is settled, not contested — a later reader
should treat the review's D5-intraday remedy as superseded by this entry.

**Decision.** `ema8/15/21_m5` are computed on the **entry session's own RTH
bars**, and are null until the session has at least `span` completed bars. The
session bound is enforced inside `compute_intraday_features` (which now filters
to `session.rth_open_at <= interval_start < session.rth_close_at`) rather than
trusted to the caller. M15/M30 have no champion — they are LD-23's new ground —
and follow the same stated convention.

**Why.** The 2026-08-04 review recorded D5's intraday half as "production's M5
EMAs run on BounceBot's '5 D' frame" and asked for a multi-session seed. Read
against the champion, that premise is wrong: `bounce_bot_lib/legacy.py` fetches
`durationStr="5 D"`/`useRTH=1` for the *previous-day extremes and the
dynamic/EOD VWAPs*, but computes the EMA levels the detector actually uses on
`today_df` alone — the code is commented "5. Calculate short EMAs (today
only)" — and leaves each level `None` unless `len(today_df) >= span`. Seeding
the warehouse column on a five-day frame would therefore have *introduced* the
same-name-different-number defect D5 exists to prevent, in the opposite
direction. What was genuinely missing is the champion's minimum-bar guard: the
warehouse published an EMA-21 built from four bars of mostly seed under the
champion's name. That guard is the real repair, and the session bound is now
structural instead of incidental.

**Rejected.** Following the review's stated remedy (a "5 D" EMA seed) — it
contradicts the champion and plan.md sec 5's "champion math is called, never
re-derived"; the review is a fallible artifact and this entry recorded the
disagreement with its evidence, which the trader then confirmed. Publishing a
short-frame EMA with a `bars_used` qualifier — the frozen sec 7.1 schema has no
such column, and a null the consumer must handle beats a number it will trust.

**Reopens if.** BounceBot's own EMA frame changes (then this follows it, under
a `feature_set_version` bump), or a registered study wants a multi-session
intraday EMA — which would be a **new, differently named** column, never a
redefinition of these three.

**Where.** `features.py::compute_intraday_features` / `ema_series(min_bars=)`;
`tests/test_warehouse_features.py::test_the_intraday_ema_lookback_is_the_session_and_needs_span_bars`
/ `test_the_intraday_ema_matches_the_champions_own_computation`.

## BD-60 — Backfill dedupes per bar, waits on a real clock, and closes the gaps it fills

**Decision.** Four changes to `run_backfill`, which together make the nightly
ETH job the thing LD-03 describes:

1. **"Already have" means a prior *backfill*, and dedupe is per bar.**
   `archive_state` reads each month partition once and returns both the
   `(symbol, interval_start)` key set (the `bar_archive._known_bar_keys`
   pattern) and the `(symbol, day)` set that carries a `capture_mode=BACKFILL`
   row. A day is skipped only when a backfill already covered it; the tee's
   `LIVE`/`DELAYED` RTH rows never suppress the request, and the ETH answer's
   overlapping RTH bars are dropped at publish instead of duplicated.
2. **Time comes from a `clock` callable, consulted on every pacer
   interaction.** The run stamp (`now`) still provides gap `detected_at`, but
   the token bucket now sees time advance.
3. **`time_budget_seconds` + the pacer's blocking `acquire`.** A run may be
   given a wall-clock budget to wait for slots; a single wait is capped at the
   pacer's own window, because an exhausted bucket refills after at most one
   window. The default is 0 (never block), so every existing caller keeps its
   behaviour and only a job that opts in waits.
4. **Honest gap reasons, deduped, and resolvable.** Pacer denials and run-cap
   exhaustion record `TIMED_OUT`; `NOT_COLLECTED_BY_POLICY` is reserved for
   genuine policy absence. `_record_missed` skips a miss whose gap is already
   open, and `resolve_gaps` closes a filled gap by appending a superseding row
   carrying `resolved_at`/`resolution=BACKFILLED`. `open_gap_keys` is the
   reader's view (latest `detected_at` per `(symbol, timeframe, gap_start)`,
   unresolved only) and `coverage_readout` now uses it.

**Why.** `already_captured` treated *any* bar for `(symbol, day)` as "already
have". The tee archives RTH bars for the whole watchlist cohort every session,
so the ETH-inclusive nightly job skipped exactly the symbols it exists to
extend, and premarket/postmarket bars for the tee cohort were never captured at
all — LD-03 defeated in full (review defect D6). Separately, `run_backfill`
computed one `stamp` and passed it as `now` to every pacer call, freezing the
10-minute window: after roughly 15 grants every remaining pair was denied and
marked missed, so one invocation could never approach sec 5.1's ~350-request
nightly plan, and `pacer.acquire` — which exists precisely to wait — had no
caller (review defect D7). Recording those denials as
`NOT_COLLECTED_BY_POLICY` conflated a pacing shortfall with intended absence,
which sec 5.4 forbids in as many words. And with no dedupe and no resolution
path, `collection_gap` grew by a row per run per miss and never recorded that
anything had been fixed.

**Rejected.** Keeping the day-level skip and adding a separate ETH-only dataset
— it splits one timeframe's bars across two datasets and breaks every reader.
Per-bar dedupe *without* the backfill-day marker — correct but it re-requests
every covered day forever, spending IB budget to learn nothing. Sleeping inside
`try_acquire` — the pacer must stay a pure arbiter; the job owns its own time
budget. Mutating the original gap row — impossible in an immutable lake; hence
supersession by `detected_at`, matching BD-53.

**Reopens if.** Pilot measurement shows the nightly budget wants a per-symbol
rather than per-run time budget, or a second writer makes the single-pass
`archive_state` snapshot stale within a run (it is refreshed per run today
because there is exactly one writer, LD-01).

**Where.** `backfill.py::run_backfill` / `archive_state` / `open_gap_keys` /
`resolve_gaps` / `_record_missed`; `queries.py::coverage_readout`;
`tests/test_warehouse_backfill.py::test_the_eth_job_still_runs_for_symbols_the_tee_already_captured`
/ `test_one_run_gets_far_past_a_single_pacer_window`
/ `test_a_frozen_run_stamp_would_have_capped_the_run`
/ `test_a_repeated_miss_does_not_inflate_the_gap_table`
/ `test_a_later_run_that_fills_a_gap_resolves_it`.

## BD-61 — The EOD build's step list, in dependency order

**Decision.** `run_build` runs fourteen steps, in this order and for these
reasons: **reconcile** and **spool seal** first, so the lake is consistent and
the session's spooled M5 bars are in it before anything reads bars;
**bronze**, because the D1 wrap, the universe snapshots and the anchors all
read wrapped artifacts; **daily snapshots** (universe membership + level
geometry), which fix the session's point-in-time cohort; **`bar_d1`**
(`ingest_daily_bars`) over that cohort, because sessions, aggregates and every
feature snapshot read it; **sessions**, **derived** and **weekly**;
**`anchor_instance`** from the bronze earnings anchors, because a daily
snapshot's AVWAP block needs an anchor to key on; **daily** then **intraday
feature snapshots**; **outcomes**; **backups**; and **retirement** last, so
nothing is swept before it has been copied.

Three sub-decisions inside that list:

* **Cohort provenance.** The D1 wrap's symbol list comes from the session's own
  `universe_membership_daily` rows, not from today's watchlist files - LD-05
  makes that snapshot the point-in-time truth, so a rebuild months later sees
  the same cohort.
* **Anchor scope.** `anchors_from_bronze` groups every version of
  `earnings_avwap_anchors.csv` bronze has seen by ticker; the newest distinct
  `anchor_date` is `EARNINGS_CURRENT` and the one before it
  `EARNINGS_PREVIOUS` (LD-09's slice scope). Older dates are history, not slice
  anchors, and a ticker seen once simply has no previous anchor.
* **Backups are opt-in by path.** `research_backup_class_a_dirs` and
  `research_backup_class_b_dir` (env: `TRADINGBOTV3_RESEARCH_BACKUP_A`/`_B`)
  gate the two backup steps. Unset, each step reports `NO_TARGET` and names the
  setting to fill in. A backup written to a guessed destination is not a
  backup, and Class B must be a second physical disk, never Drive (sec 8.5).

Occurrence ingestion now runs immediately before outcomes through BD-44's
tracker adapter. When no eligible tracker rows exist, the outcomes step still
reports `NO_OCCURRENCES` honestly. Its
`bands_by_occurrence` is pinned to each occurrence's own trigger-session
`feature_snapshot_daily` row, which is the review's point-in-time requirement:
bands computed later than the trigger would be look-ahead.

**Why.** `run_build` stopped at derived/weekly. It never called
`ingest_daily_bars`, the anchors, either feature snapshot, the outcome engine
or `backup_class_a/b` — so as shipped, a night of capture produced no
`bar_d1`, no features, no outcomes and no backup at all, and the pilot would
have accumulated 20 sessions of raw bars with nothing built on them. BD-20 and
BD-44 declare their gaps; this one was undeclared (review defect D19).

**Rejected.** Running the nightly/weekly *backfill* jobs from this build too —
they are net-new provider traffic on a different schedule (overnight, after
the TWS restart) and belong to their own invocation with its own time budget
(BD-60); folding them in would make one job both a fast EOD build and a
multi-hour fetch. Guessing backup destinations from the Drive home folder —
Class B explicitly must not live there.

**Reopens if.** The pilot shows the EOD build's wall time needs the feature pass
split from the seal.

**Where.** `cli.py::run_build` / `cohort_for` / `anchors_from_bronze` /
`anchor_dates_by_symbol` / `_run_outcomes` / `_bands_by_occurrence` /
`_run_backups`; `config.py::backup_class_a_dirs` / `backup_class_b_dir`;
`tests/test_warehouse_restore.py::test_the_build_job_runs_the_whole_step_list`
/ `test_backups_no_op_with_a_clear_message_when_unconfigured`
/ `test_the_anchor_step_reads_current_and_previous_from_bronze`.

## BD-62 — Five edge-case repairs: orphan adoption, year boundaries, empty episodes, naive IB strings, gap counts

**Decision.** Five independent hardening choices, grouped because each is small
and none changes an interface:

1. **Reconcile refuses an orphan that overlaps live rows (D14).**
   `_overlaps_live_rows` compares the orphan's rows against the partition's
   live rows at the **dataset's declared grain**, not by file hash, and
   quarantines under `ORPHAN_OVERLAPS_LIVE_ROWS` when they intersect.
   Unreadable live state also refuses. BD-03's adopt-don't-discard survives
   intact for the publish-retry case it was written for.
2. **`latest_occurrences` reads adjacent years (D15).** It now spans
   `year ± span_years` (default 1). `setup_occurrence` partitions on
   `event_at`, and a revision carries the original `event_at` forward, so the
   rescan's own year is not where its predecessor lives.
3. **`build_occurrence_row` rejects a detection with no episode identity
   (D16).** Neither `anchor_instance_id` nor `episode_start` means no row, and
   the caller counts it as `INCOMPLETE_DETECTION`.
4. **`_epoch_to_utc` accepts only epoch seconds and already-aware datetimes
   (D17).** The naive-string fallback is gone rather than re-zoned, and a naive
   `datetime` returns `None` too. Additionally the epoch parse now range-checks
   its result: `"20260803"` is all digits and was being read as epoch seconds,
   landing the bar in **1970-08-23** — found by the regression test for this
   entry, not by the review.
5. **`collection_gap.expected_bars` holds the expected count (D18).** Because
   `gap_start`/`gap_end` span the whole session, the honest value for that
   interval is the session's expected bar count; the per-run shortfall moved to
   `GapReport.missing_bars_by_reason`.

**Why.** (1) A compaction crashing between its `os.replace` and its manifest
append leaves a merged file whose hash matches nothing registered — so the
hash guard waved it through — while its source parts stay live; adopting it
double-counted every row in the partition, and the next compaction balanced
because both sides doubled. (2)/(3) Episode counts are the denominator of every
evidence floor, so a year-boundary duplicate or a permanent two-theses-one-id
collapse corrupts the arithmetic silently. (4) Reading exchange-local strings
as UTC shifts bars 4-5 hours with no signal. (5) The Health coverage tile summed
a column whose name and content disagreed.

**Rejected.** For D14, a pre-intent marker in `_incoming/` before compaction —
also valid (the review offered both) but it adds a fifth step to a locked
4-step seal and only protects compaction, whereas the row-overlap test
protects any future writer that can produce an overlapping orphan. For D18,
narrowing `gap_start`/`gap_end` to the missing bars — that needs per-interval
gap detection this build does not do, and inventing an interval is worse than
an honest session-wide one.

**Reopens if.** A dataset appears whose grain does not identify a row (then
D14's check needs a content hash per row), or per-interval gap detection lands
(then D18's interval narrows and `expected_bars` follows it).

**Where.** `store.py::_overlaps_live_rows` / `_grain_keys` /
`QUARANTINE_ORPHAN_OVERLAPS_LIVE`; `occurrences.py::latest_occurrences` /
`build_occurrence_row`; `ib_capture.py::_epoch_to_utc`;
`bar_archive.py::record_collection_gaps` / `GapReport.missing_bars_by_reason`;
`tests/test_warehouse_seal.py::test_a_crashed_compaction_is_quarantined_not_adopted`
/ `test_a_genuinely_new_orphan_is_still_adopted`;
`tests/test_warehouse_occurrence.py::test_a_december_occurrence_rescanned_in_january_revises_not_duplicates`
/ `test_a_detection_with_no_episode_anchor_is_rejected_not_collapsed`;
`tests/test_warehouse_backfill.py::test_a_naive_ib_timestamp_is_dropped_not_rezoned`;
`tests/test_warehouse_tee.py::test_policy_absence_is_never_recorded_as_missing`.

## BD-63 — The live tee is a GUI-owned capture object on its own 60s timer

**Decision.** `WarehouseTeeCapture` (in `ui/services/warehouse_service.py`) owns
the live M5 tee. `BounceService` constructs it **lazily, only when a bot exists
and `warehouse_enabled()` is true**, drives it from a service-owned 60-second
`QTimer` armed on `started` and stopped with every other timer on shutdown, and
calls it from the `capture_warehouse_tee` slot — on the GUI thread, which is the
thread that owns `bot.latest_bars`. The object takes `dict(bot.latest_bars)`
itself and calls `capture_m5_tee(None, snapshot, spool=writer, seen=session_set)`:
`store=None` means **zero lake I/O on the GUI thread**, not even the read that
normally seeds de-duplication — its own per-session `seen` set does that, which
is exactly what the `seen` parameter exists for. Any exception is logged once
and swallowed. Health renders `warehouse_health_tiles` as six check rows,
computed on the page's existing audit worker thread; `OFF` maps to UNKNOWN, and
a red tile can worsen the page's verdict but a green one never improves it.

**Why.** BD-20/BD-52's open item: the tee, the spool, the seal and the tiles all
existed and nothing called them, so the pilot could not start. The review's
design ruling fixed the shape (service-layer, never inside `bounce_bot_lib`,
snapshot on the owning thread, spool-only) and this implements it literally.
60 seconds rather than the existing 3-second health cadence because M5 bars
complete every five minutes — anything faster just re-scans the same dict — and
rather than folding into `refresh_health` because a timer the service owns
outright is easier to reason about and to stop.

**Rejected.** Calling the tee from inside BounceBot's own cycle — it would put a
warehouse import on a champion path and make a capture failure a champion
failure. Passing a live `ResearchStore` and letting `capture_m5_tee` dedupe
against the lake — correct offline, but that is a parquet read on the GUI
thread every minute. Snapshotting inside `extract_tee_bars` — too late; the
resize can already have happened.

**Reopens if.** The post-slice Focus-streaming milestone lands (capture then
issues real requests and the pacer's champion-observation question from the
review's section 1 reopens with it).

**Where.** `ui/services/warehouse_service.py::WarehouseTeeCapture`;
`ui/services/bounce_service.py::capture_warehouse_tee` /
`_start_warehouse_timer`; `ui/panels/health_panel.py::warehouse_checks` /
`_with_warehouse_checks`; `tests/test_qt_warehouse_tee.py`.

## BD-64 — A gap's interval is the session's own window, and `expected_bars` counts it

**Decision.** `_gap_window(day, use_rth=, interval=)` returns the exchange
session's boundaries for the scope the job actually requested — ETH
(`useRTH=0`, LD-03) for the nightly M5/M1 capture, RTH otherwise — together
with the bar count expected across that interval. `collection_gap.gap_start`/
`gap_end`/`expected_bars` are written from it, and `_record_missed` and
`resolve_gaps` both key on it so the dedupe and resolution keys stay in step.
A day the exchange was closed has no interval to name and expects nothing: it
keeps the calendar day and a count of **0**, because a holiday is an honest
absence, not a shortfall.

The count has exactly one definition, and it is the calendar's:
`TradingSession.window(extended=)` and `TradingSession.expected_bars(minutes,
extended=)` derive from the session's own boundaries, and the existing
`expected_m5_bars_rth`/`expected_m1_bars_rth` properties are now expressed
through them. No constant is duplicated and no schema column is added.

**Why.** BD-62/D18 moved `expected_bars` to mean "the count expected across the
gap interval", but `backfill._record_missed` still wrote a hard-coded `0` while
`gap_start`/`gap_end` spanned a whole UTC calendar day — an interval that is
neither RTH nor ETH and a count that contradicted the column's stated meaning
(review defect D20). The Health coverage tile reads these rows, so a
permanently-zero expectation makes "expected vs observed" unanswerable for
exactly the job LD-03 puts inside the slice.

**Rejected.** A `TIMEFRAME_EXPECTED_BARS` lookup table — a second definition of
"expected", which is how the two numbers with one name start disagreeing;
the job already carries its `interval`, so the count is derivable. Keeping the
calendar-day interval and computing an ETH count for it — the interval and the
count would still disagree, which is the defect.

**Reopens if.** A capture scope other than RTH/ETH is added (the `extended`
flag becomes a scope enum), or half-day sessions need a distinct contract —
they already work, because the boundaries come from the session.

**Where.** `exchange_calendar.py::TradingSession.window` / `expected_bars`;
`backfill.py::_gap_window` / `_record_missed` / `resolve_gaps` / `run_backfill`;
`tests/test_warehouse_backfill.py::test_gap_rows_carry_the_intervals_expected_bar_count`
/ `test_an_rth_scoped_job_expects_the_sessions_rth_bars`
/ `test_a_paced_out_gap_is_timed_out_and_still_counts_its_interval`
/ `test_a_closed_exchange_day_expects_no_bars`
/ `test_supersession_still_resolves_after_the_interval_change`.

## BD-65 — The GUI tee submits a snapshot; a worker thread does every byte of I/O

**Decision.** `WarehouseTeeCapture` splits in two. `submit(bot)` is what the
Qt slot calls and is **pure memory**: `dict(bot.latest_bars)` into a one-slot
mailbox, a lock, an event — no filesystem, no config read, no warehouse import.
A worker thread owned by the capture object does everything else: it checks
`warehouse_enabled()` once, builds the `ResearchSpoolWriter`, and drains the
mailbox. The mailbox is **latest-wins, never a queue** — a tick landing while
the worker is busy replaces the pending snapshot, which cannot backlog and
loses nothing, because the champion's cache is a rolling five-day window and
the next tick re-offers the same bars. `close()` idles the worker and is
**reversible**: a later `submit` resumes the same object.

**Why.** The BD-63 wiring correctly avoided provider requests and lake I/O but
still did local filesystem work on the 60-second GUI timer (review defect D21).
`ResearchSpoolWriter.__init__` creates its directory and adopts *every* stale
`.open` segment it finds, renaming each — unbounded in the number of segments a
crashed session left behind. And it is not only construction: every `write`
runs `enforce_cap`, which globs the spool, stats each segment and may *read*
segment contents to classify them, and then `fsync`s. On a DAS hiccup or a
slow disk that is a GUI stall, on the desk's own trading surface, for research
evidence.

The restart semantics are load-bearing, not cosmetic: `ResearchStore.publish`
does **not** dedupe on grain — that is the caller's job, which is what
`capture_m5_tee(seen=...)` exists for — so a Stop/Start that built a fresh
capture object would re-spool the whole session and seal duplicate `bar_m5`
rows. Hence one object with one `seen` set across the restart.

**Rejected.** Bounded init plus deferred adoption (the review's option 2) —
it fixes construction and leaves `enforce_cap` and `fsync` on the GUI thread,
which is the same defect at a smaller size. A `QThread`/`QThreadPool` — the
capture holds no Qt object and must survive independently of the widget tree.
An unbounded queue — a stalled disk would grow it without limit, and every
queued snapshot is stale by construction anyway.

**Reopens if.** The tee ever needs to write something the champion's cache does
not already hold (then the mailbox's latest-wins assumption no longer holds and
it becomes a real queue with a stated bound).

**Where.** `ui/services/warehouse_service.py::WarehouseTeeCapture.submit` /
`close` / `wait_idle` / `_run` / `_capture_snapshot`;
`ui/services/bounce_service.py::capture_warehouse_tee` /
`_close_warehouse_capture`; `tests/test_qt_warehouse_tee.py` (the D21 block).

## BD-66 — A bar belongs to its exchange session, never to its UTC date

**Decision.** `archive_state` buckets the "already backfilled" marker by the
row's `session_id` — a frozen sec-7.1 column the writers already populate
correctly — falling back to `session_context(stamp).session_date` only if a row
somehow lacks one. It also reads the **following month's** partition for every
requested day, because a session's own ETH tail can live there.

**Why.** ETH runs to 20:00 ET, which is 01:00 UTC the next day under EST, so
the final hour of every winter session is stored under tomorrow's UTC date.
Bucketing by `interval_start.date()` was wrong in both directions at once
(review defect D22, reproduced on 2026-01-14): the session that *was*
backfilled went unmarked, so the nightly job re-requested it forever and never
converged; and the *following* session was marked covered although nothing had
collected it, so its request was skipped outright and its ETH bars were never
captured. LD-03 says uncaptured ETH history is permanently lost, which makes
that second half data loss, not inefficiency — S2, not the S3 it was filed as.
The month-partition half is the same crossing one level up: 31 December's
19:30 ET bars are 1 January in UTC, so the per-bar dedupe could not see them
and republished them.

**Rejected.** Deriving the session date from the timestamp everywhere and
ignoring `session_id` — it re-derives a fact the row already carries, which is
how a second answer to one question appears. Widening the partition scan
unconditionally — the next day's month is enough, because ETH opens at 04:00 ET
(09:00 UTC) and never reaches back into the previous day.

**Reopens if.** A capture scope opens earlier than 04:00 ET (then the previous
month's partition joins the read), or a writer is added that does not populate
`session_id`.

**Where.** `backfill.py::_session_date_of` / `_bar_partitions` /
`archive_state`;
`tests/test_warehouse_backfill.py::test_a_winter_eth_tail_marks_its_own_session_not_the_next_one`
/ `test_a_month_crossing_eth_tail_is_not_republished`.

## BD-67 — A gap is closed by any run whose fetch covered its interval

**Decision.** `resolve_gaps` matches by **interval containment** rather than by
an exact `gap_start` key: a run closes every open gap for the same
(symbol, timeframe) whose `[gap_start, gap_end]` lies inside the window it just
fetched. Containment is directional, so an RTH-scoped run does not close a
wider ETH gap.

**Why.** Found while re-checking gap supersession for BD-64. Different jobs
record the same session's absence at different scopes: `record_collection_gaps`
(the tee's session audit) writes at `rth_open_at`, while the nightly backfill
works the ETH window. Exact-key matching meant a tee-recorded gap could
**never** be closed by the backfill that actually filled its bars — it stayed
open forever and the Health coverage tile went on reporting a session short
that was not. This predates BD-64 (the old midnight-UTC key matched
`rth_open_at` no better), but BD-64 is where the two intervals became
explicit enough to see it.

**Rejected.** Keying every writer at the same instant so exact matching works —
it would force the tee to describe an ETH interval it never inspected, which
is the D18 defect in a new place. Closing every open gap for the session
regardless of interval — an RTH-only run would then silently claim the ETH
window it never fetched.

**Reopens if.** A capture scope appears that is neither contained in nor
containing the others (then containment needs a scope lattice rather than a
comparison).

**Where.** `backfill.py::resolve_gaps`;
`tests/test_warehouse_backfill.py::test_a_backfill_closes_the_tee_recorded_gap_it_filled`
/ `test_an_rth_only_run_does_not_close_a_wider_eth_gap`.

## BD-68 — The compaction-orphan guard exempts datasets that supersede by time

**Decision.** `store.SUPERSEDING_DATASETS` = `{collection_gap, outcome_path}`
is exempt from BD-62/D14's overlap refusal: for those two, an orphan sharing a
grain key with a live row is adopted normally.

**Why.** Found while re-checking `_overlaps_live_rows` against the registry.
The D14 guard assumes a repeated grain key means duplication, which holds only
where the writer publishes one row per grain. `outcome_path`
(`occurrence_id, recipe_id, outcome_definition_id`) supersedes by `computed_at`
under BD-53 and `collection_gap` (`symbol, timeframe, gap_start`) by
`detected_at` under BD-60/BD-67 — neither carries a revision column *inside*
its grain, so a recomputed outcome or a gap resolution shares its predecessor's
key **by design**. Without the exemption, a publish that crashed between its
`os.replace` and its manifest append would have had that legitimate row
quarantined instead of adopted. Self-healing (the next build recomputes, and
quarantine preserves the row), so S3 — but it silently discards a night's
recomputation and inflates the quarantine count the Health tile watches.

Datasets that *do* discriminate revisions in the grain — `setup_occurrence`
(`revision_id`), `anchor_instance` (`system_from`), `bar_m5`/`bar_d1`
(`revision_id`) — need no exemption and stay guarded. So do the several
datasets whose grain lacks a discriminator but whose builders publish one row
per grain and skip what exists (`bar_derived`, `trading_session`,
`scan_coverage`, `universe_membership_daily`, both feature snapshots).

**Rejected.** Deriving the exemption from the schema — it is not derivable:
whether a repeated grain means duplication is a property of the *writer*, and
guarded and exempt datasets look identical in the registry. Comparing full row
content instead of the grain — precise, but it reads every column of a live
partition where the grain projection reads four, and the compaction-crash case
it must catch is exactly the one where content is identical.

**Reopens if.** A new dataset supersedes by time without a revision column in
its grain — it must be added to the set by hand. The pinning test states this.

**Where.** `store.py::SUPERSEDING_DATASETS` / `_overlaps_live_rows`;
`tests/test_warehouse_seal.py::test_a_superseding_orphan_is_adopted_not_quarantined`
/ `test_the_supersession_exemption_is_a_deliberate_hand_maintained_pin`.

## BD-69 — Outcome simulation reads each occurrence's own month, not the build day's

**Decision.** `_m5_partitions_for` builds the M5 read set from the *trigger
month of every occurrence being simulated* (plus the following month for the
BD-66 ETH tail, plus the build day's own month), instead of the build day's
month alone.

**Why.** Found while re-reading `_run_outcomes` as if BD-44 had landed, per the
review's instruction. `known` spans two years of occurrences and BD-53
re-simulates every **non-terminal** one on every build — but the M5 read was
`month={build day}`. An intraday occurrence triggered in any earlier month was
therefore handed an empty bar list every night and drew its conclusion from
that absence: it would churn an `OPEN` row to `TRUNCATED`, or write a
superseding row asserting a truncation that never happened. Silent, nightly,
and exactly the "missing data is uncertainty, never confirmation" invariant
(plan.md sec 5) inverted.

**Rejected.** Reading every month in range — unbounded as the corpus grows,
for no gain, since terminal outcomes are skipped inside `build_outcomes`
anyway. Filtering to non-terminal occurrences before choosing partitions — it
duplicates `build_outcomes`' own terminality rule outside it, which is how the
two drift.

**Reopens if.** The occurrence corpus grows large enough that reading a
partition per trigger month costs real build time — the narrowing is then to
occurrences whose latest outcome is non-terminal, done *inside*
`build_outcomes` where that rule already lives.

**Where.** `cli.py::_m5_partitions_for` / `_run_outcomes`;
`tests/test_warehouse_restore.py::test_outcome_simulation_reads_each_occurrences_own_month`.

## BD-70 — Every warehouse job now has a real invoker

**Decision.** Three seams closed, so the warehouse runs rather than merely
existing:

1. **The build is invoked post-scan, in process.**
   `ScanService.start_warehouse_build` runs `run_build` on its own thread when a
   scan finishes — LD-01's "post-scan/EOD CLI build job" without a daemon. One
   at a time (a second is skipped; `run_build`'s single-flight lock refuses a
   concurrent one from any other process). Gated on `warehouse_enabled()`
   *inside* the worker, never raising into the scan, and joined on shutdown so a
   build mid-seal finishes its manifest line.
2. **The backfill jobs have a CLI entry point.**
   `cli backfill --job nightly|weekly|seed` drives `run_nightly_backfill` /
   `run_weekly_universe_sweep` / `run_yahoo_seed`, which previously had **no
   caller anywhere**. It takes the **same single-flight lock as the build** —
   both write the lake and LD-01 allows exactly one writer — sources its cohort
   from `universe_membership_daily` (LD-05 point-in-time membership, the same
   source the D1 wrap uses), and defaults the nightly wait budget to 4 h of the
   ~5.5 h overnight window (sec 5.1). A missing cohort or an unavailable
   transport is a reported *status*, never a crash.
3. **`register_build_job` describes reality.** It previously probed for a
   `scheduler.register_job` method that exists nowhere in the repository, which
   made the build look scheduled while nothing ran it. It is now a descriptor
   naming its real invoker, and a test asserts that invoker exists.

**Why.** Capture, sealing, features and outcomes were all implemented and
individually tested, and *none of them ran*. The tee spools M5 bars every
minute and only the build seals them; because M5 segments are `PROTECTED` and
never shed (LD-12/BD-18), an unsealed spool does not self-limit — it grows until
Health goes red. So the missing invoker was not a convenience gap, it was the
difference between the pilot capturing evidence and the pilot filling a disk.
The nightly ETH backfill is inside the slice by LD-03, so it needed an entry
point on the same footing.

**Rejected.** A daemon or a timer of the warehouse's own — LD-01 is explicit
that there is none, and the scan-completion point already exists. Running the
build inside `run_build`'s caller on the GUI thread — it seals, wraps bronze and
computes features; that is exactly the D21 mistake one level up. Giving backfill
its own lock — two lake writers, which LD-01 forbids.

**Reopens if.** A real scheduler is added to the repository (the descriptor is
then handed to it), or measurement shows the post-scan build overruns the gap
between scans (it would move to EOD-only).

**Where.** `ui/services/scan_service.py::start_warehouse_build` /
`_run_warehouse_build` / `wait_for_warehouse_build`;
`cli.py::run_backfill_job` / `_run_ib_backfill` / `_run_seed` /
`_backfill_cohort` / `main`; `ui/services/warehouse_service.py::register_build_job`;
`tests/test_qt_warehouse_tee.py` (post-scan block),
`tests/test_warehouse_restore.py` (backfill entry-point block).

## BD-71 — A shed spool segment is an open gap with no resolution

**Decision.** `_seal_shed_log` writes `resolution=None` alongside its null
`resolved_at`, instead of `resolution="POLICY"`.

**Why.** The row asserted both states at once: a `resolution` says the gap was
settled, a null `resolved_at` says it is still open. Since BD-60 the open-gap
view keys on `resolved_at is None`, so these rows *are* open — and a shed M5
window is genuinely still recoverable from the provider, so leaving it open is
the honest reading. Carrying a resolution string it never earned would also have
excluded it from `resolve_gaps`' containment closure (BD-67) if a later
backfill did refill that window.

**Where.** `spool.py::_seal_shed_log`;
`tests/test_warehouse_spool.py::test_shed_evidence_becomes_an_explicit_gap_row`.

---

## BD-72 — The home-folder refusal survives the death of Google Drive

**Decision.** `config._refuse_shared_home` is kept exactly as written now that
`C:\TradingBotData` is a plain local folder rather than a Drive-synced one
(decision 0015). The lake was pointed at `\\MINI-PC\Trading Bot Data\research_lake`
on 2026-08-10; the spool stays machine-local at
`%LOCALAPPDATA%\TradingBotV3\research_spool`.

**Why.** The guard's original rationale — Drive quota, sync locks, DriveFS
wedges — evaporated with the sync client, so it was worth asking whether the
guard should go too. It should not. Two independent reasons survive, and either
alone justifies it: the home folder is the *compact operational* storage class
and a lake inside it destroys that distinction; and `push_cold_to_das.ps1`
mirrors home-folder subtrees wholesale, so a lake living there would be copied
to the DAS a second time, by a different mechanism, with no manifest.

The spool location is not configurable and should stay that way — it is the
local-first staging step, and putting it on the DAS would defeat its purpose of
surviving a file-server outage.

**Where.** `research_warehouse/config.py::_refuse_shared_home`,
`research_spool_dir`; `docs/decisions/0015-no-cloud-sync-das-file-server-storage.md`.

**Reopen if:** the home folder itself moves onto the DAS, or the cold-push
script is narrowed to an explicit allowlist that could safely exclude a lake.

---

## Open items for Sol / Fable

Things this build deliberately left for a human decision or a live check.
Each is already stated in its own BD entry; this is the short list.

| # | Item | Where | What is needed |
|---|---|---|---|
| 1 | ~~Nothing calls the tee during a live session.~~ **Wired** (BD-63, BD-70): the tee runs on a 60s timer, the build runs post-scan, and the backfill has a CLI entry point. The live path still needs desk observation against real BounceBot/TWS data. | BD-20, BD-63, BD-70 | Watch the tiles on the first live session, then start the pilot |
| 1b | **The 20-session pilot has not run** — it is a live-desk activity, not code. | BD-52 | Run it once capture is live; log the sec 5.6 measurements |
| 2 | **`ib_capture.build_ib_transport` is unverified** — the real ibapi client has no offline test and no broker-marked live run. | BD-25 | One live run on the desk before the pilot leans on it |
| 3 | **`exploration_cohort.txt` is empty** — the fixed 30 symbols define part of the research denominator, so no agent invented them. | BD-12 | Trader supplies the list (confirmation register item 5) |
| 4 | **Two favorite-zone definitions are the builder's** — `first_dev_touch_order` and `band1_rejection_strength`. | BD-32 | Confirm or amend; a change is a `feature_set_version` bump |
| 5 | **Production context columns are null until Phase 6** wires the bounce-ledger join. | BD-33 | Nothing now; noted so the gap is not mistaken for "no signal" |
| 6 | **DYNAMIC and EOD session VWAP are not yet captured** — only STANDARD. | BD-34 | Wrap the other two champion paths when their consumers need them |
| 7 | **Unscheduled exchange closures** cannot come from calendar rules; they appear as sessions with no bars. | BD-26 | Add a dated override list if one ever occurs |
| 8 | ~~No detector adapter yet.~~ **Closed 2026-08-27:** the transition-ledger/scenario adapter covers all 16 canonical tracker families without parsing the giant snapshot. | BD-44 | Live canary only; explicit BounceBot linkage remains separately under BD-43 |
| 10 | ~~DuckDB desktop verification.~~ **Closed 2026-08-09:** `duckdb==1.5.5` was installed in the uv-managed Windows Python 3.12 environment and the full desk suite passed. | BD-45 | None; DuckDB remains optional and read-only |
| 9 | **Bounce link is a time window** (symbol + session + ±60 min), not an explicit key. | BD-43 | Confirm the window, or add an occurrence link to the bounce ledger |

---

## Standing constraints this build re-checks every phase

Not decisions — the invariants each phase is audited against, listed so a
reviewer can spot-check them quickly:

1. Shadow-only: no detector, score, ranking, alert, or champion-timing path
   imports the warehouse package.
2. Total no-op when `research_store_dir` is unset (`ResearchStore.open()` →
   `None`, every entry point returns early).
3. The lake never lives inside the `C:\TradingBotData` home folder (config refuses
   it). Since decision 0015 that folder is plain local storage, not Drive-synced;
   the refusal stands on storage-class and cold-push grounds. See BD-72.
4. Completed bars only; missing data is uncertainty, never confirmation.
5. One owner per job; a failed publish never destroys the last verified artifact.
6. `calc_anchored_vwap_bands` is wrapped and parity-tested, never reimplemented
   (Phase 5).
7. The shared IB pacer never delays or queues champion traffic, and capture
   errors never count against the champion Yahoo circuit breaker (Phase 3/3b).

## BD-XX — reconcile tolerates the system clock being coarser than the filesystem

**Date:** 2026-08-15 (R2.1 item 6, trader-approved before the edit)

**Decision.** `WarehouseStore.reconcile` widens its incoming-file cutoff by
`CLOCK_GRANULARITY_SECONDS = 0.05` so a file cannot be judged "newer than now"
by clock resolution alone.

**Why.** Windows' system clock ticks about every 15.6 ms while NTFS stamps
mtimes far more finely, so `utc_now()` could round *below* the mtime of a file
written microseconds earlier. That file then failed the `st_mtime > cutoff`
test and was never quarantined. Invisible against the 3600 s default grace, but
with `incoming_grace_seconds=0` it made the outcome a coin flip — measured at 3
failures in 6 runs on the desk, reproducing in isolation, which had previously
been misdiagnosed as load-related flakiness.

**Why a fix rather than a test quarantine.** The behaviour was genuinely wrong,
not just awkward to test: a file written in the same clock tick as a zero-grace
check *is* stale and should be quarantined. Quarantining the test would have
kept the defect and lost the coverage.

**Scope.** Widens the quarantine window by 50 ms. Inconsequential beside any
real grace period, so production behaviour is unchanged.

**Reopen if.** A platform is added whose filesystem timestamps are coarser than
its clock, or a caller needs sub-50 ms grace semantics.


## BD-73 — Very large SNAPSHOT payloads are stored whole but not `json.loads`-ed

**Date:** 2026-08-27 (desk-memory packet, trader-authorised build prompt).

**Decision.** `ingest_existing.SNAPSHOT_PARSE_MAX_BYTES = 64 MB`. Above it a
`MODE_SNAPSHOT` payload is captured in full and published unchanged, but is not
parsed; `parsed` becomes `{}` when `_looks_like_json` (first and last non-space
characters form a `{}`/`[]` pair) holds, and `None` when it does not, which is
what still drives `quality` to `COMPLETE` / `INVALID_DATA`.

Separately and unconditionally, `ingest_artifact` now hashes the source with
`_sha256_path` (chunked) and answers the watermark BEFORE `read_bytes`, so an
UNCHANGED verdict costs no allocation at all.

**Why.** `master_avwap_setup_tracker.json` measured **1,026,057,028 bytes** on
2026-08-27. The old order read it whole and hashed the bytes *before* comparing
the watermark, so every bronze ingest allocated 1.03 GB inside the desk process
— including the ~90% that immediately concluded UNCHANGED. When the sha had
changed, `json.loads` over the decoded text added several GB more, on top of
the warehouse build's own peak, and the desk was measured at 8–13 GB.

**Why the skip loses nothing here, stated precisely.** The parse feeds exactly
three things: `_parse_event_at(parsed, artifact.event_keys)`,
`_first_value(parsed, artifact.id_keys)`, and the `quality` flag. The
`setup_tracker` artifact declares **neither** `event_keys` nor `id_keys`, so
`_parse_event_at` returns `None` on its first line (`if not keys`) and
`_first_value` returns `""` without inspecting the payload — parsed or not.
For this artifact the parsed row and the skipped row are byte-identical, and a
regression test asserts it rather than trusting the reasoning.

**What was rejected.** (a) Marking an unparsed row `INVALID_DATA`: false, and
it would poison any downstream quality filter. (b) Streaming the parse: the
result would still be several GB of dicts, which is the cost being removed.
(c) Not storing the payload above the threshold: that changes the bronze
contract and loses the artifact, which is the one thing bronze exists to keep.
(d) Widening `QUALITY_STATES` with a new "STORED_UNPARSED" state: it would be
a schema-visible change for a distinction no reader currently makes.

**Residual, not hidden.** A changed snapshot still costs roughly `size` bytes
plus a same-size decoded `str`, because `payload_text` must be a Python string
for the publish path. ~2 GB for the tracker, down from several. Removing that
too means changing the bronze publish path, which this packet deliberately did
not touch.

**Reopen if.** (1) `setup_tracker` — or any artifact that can exceed 64 MB —
gains `event_keys` or `id_keys`: the skip would then silently empty real
columns and the threshold must be revisited, not merely raised. The test
`test_bronze_snapshot_large_files.py::tracker_artifact` fails loudly in that
case. (2) A reader starts depending on `quality == COMPLETE` meaning "fully
parsed" rather than "captured whole and JSON-shaped". (3) The bronze publish
path learns to stream, at which point the threshold can rise or disappear.

## BD-74 — Session/symbol narrowing belongs in Arrow, in one store helper

**Date:** 2026-08-27 (same packet).

**Decision.** `ResearchStore.read_rows(dataset, partition, *, columns, symbols,
interval_start_range, occurrence_ids, recipe_ids)` filters through
`Dataset.to_table(filter=...)` before
`to_pylist()`, and the three build steps that read `bar_m5` use it:
`aggregate.build_derived_bars`, `features.build_intraday_snapshots` and
`cli._run_outcomes`.

**Why.** Partitions are MONTH-keyed while these steps each want one session (or
one symbol set). `read_table(partition).to_pylist()` therefore materialised the
whole month as Python objects so that a few percent of it could be used, and
the cost grew all month: `silver/bar_m5/month=2026-08` was **8,704,108 rows /
408 MB parquet** on 2026-08-27, `to_pylist` cost **1,769 B/row = 15.4 GB**, and
the largest single session in it was 588,778 rows — 6.8% of the month. Measured
after: 0.53 GB for a full session, 0.31 GB for a 20-symbol outcome read.

**Why a narrow helper and not a free-form filter argument.** Only the two
predicates the callers actually replaced are offered, so a future caller cannot
express something subtly different from the Python test it stands in for.
`symbols` matches EXACTLY (no case folding, no stripping) because the
`symbol in wanted` checks it replaces did; an empty sequence means no filter,
which is what the callers pass when no cohort was named.

**What was deliberately NOT narrowed.** `_run_outcomes` filters by symbol only,
never by date: the outcome walk needs ATR warm-up and runs FORWARD over a
horizon that can cross sessions, which is exactly why `_m5_partitions_for`
widens to the trigger's previous, own and following months (BD-66, BD-69,
BD-75). Narrowing it to a day would
re-simulate against a truncated future — the same class of defect BD-69 fixed.
`build_intraday_snapshots` applies the symbol filter only when the caller named
symbols, because with none named its cohort is derived from the bars present in
the session, so narrowing the read would change the answer and not just the
cost.

**Also deliberately not done.** Moving `run_build` into a child process. The
in-process single-flight lock, the spool seal and the ledger's `_record_job`
all assume one process, and the filtering removes the growth on its own. It
stays a decision, not owed work; the trader decides if it is ever wanted.

**Reopen if.** A build step needs a predicate these two cannot express, or the
partition key changes from month to something finer (at which point the helper
becomes redundant rather than wrong).

## BD-75 — D1 tracker studies enter on the next session's first completed M5 close

**Date:** 2026-08-27 (trader-directed stop/target research packet).

**Decision.** `M5_CLOSE_RECIPES` is a separate bounded research grid and does
not alter frozen `RECIPES`. Every eligible D1 tracker occurrence enters at the
next regular session's first completed M5 close. Structural stops select the
nearest valid tracker level of each source type at rank 1, 2, or 3; ATR controls
use 0.5, 1.0, or 1.5 ATR. Each is crossed with 1R, 2R, and 3R targets: 54
recipes. Same-bar ambiguity is STOP_FIRST. The existing deterministic fallback
cost model is used without reading bid/ask. No M1 data and no trader-planned
stop/risk are inputs.

**Why.** The trader asked the warehouse to discover useful stop and profit
locations, so requiring their planned stop would condition the answer on the
thing being studied. A D1 fact is known after close; the next completed M5
close is a reproducible executable proxy at the requested granularity.

**Missingness.** No next-session M5 bar, ATR, or valid stop geometry means no
invented result. At build time the durable M5 archive begins in August 2026, so
older tracker occurrences remain a visible coverage gap until backfill exists.

**Where.** `outcomes.py::simulate_m5_close_opportunity` /
`M5_CLOSE_RECIPES`; `tests/test_setup_research_pipeline.py`.

## BD-76 — Market bias is five point-in-time readings, not one daily label

**Date:** 2026-08-27 (same packet).

**Decision.** `setup_market_context` stores one row per occurrence for M5, M30,
H1, H4 and D1 under `auto_market_bias_multiframe_v1`. The live VWAP decision
and its early-session day-percent fallback now live together in the pure
champion `_auto_market_regime_stats`; both live callers and research call it,
so no threshold or fallback is copied. M30/H1/H4 are derived from completed SPY
M5 bars with the existing aggregate helper; D1 uses only prior completed daily
bars. Truly absent input writes `unknown`.

**Why.** One daily regime hides the exact cross-timeframe condition the study
is meant to measure. A versioned additive context table lets the nightly report
compare those cells without teaching any scanner a new rule. Earnings reports
and fundamentals remain out; technical earnings anchors remain ordinary setup
geometry.

**Where.** `market_bias_context.py`; `schemas.py::SETUP_MARKET_CONTEXT`;
`tests/test_setup_research_pipeline.py`.

## BD-77 — Outcome work is bucketed; the nightly model never computes evidence

**Date:** 2026-08-27 (same packet).

**Decision.** The in-process outcome step uses 32 stable symbol-hash buckets
(small cohorts of at most 64 symbols run whole), and `read_rows` now supports
Arrow-side exact occurrence/recipe filters. Each bucket publishes outcomes and
five-timeframe context. The final nightly `setup_research` slot always writes a
deterministic JSON/Markdown fact pack. A medium local model may narrate only if
at least one family/side/recipe cell has n>=30, five symbols and five entry
sessions; below that floor no model is called.

**Why.** The 54-recipe grid multiplies rows and must not recreate BD-74's
whole-dataset materialization inside the desk. The evidence floor prevents a
nightly AI from making sparse cells sound important. All arithmetic routes
through `evidence_stats`; the model sees a bounded fact pack and may explain or
name at most three next tests. It cannot write live policy.

**Where.** `cli.py::_run_outcomes`; `store.py::read_rows`;
`ai_jobs/setup_research.py`; `ai_jobs/runner.py`; AST and behavior guards in
`tests/test_setup_research_pipeline.py`.

## BD-78 — H2 is reopened, because the reopen condition it was cut on now holds

**Date:** 2026-09-01 (Phase 0.12 B1).

**Decision.** `H2` (120 minutes) joins `aggregate.TIMEFRAME_MINUTES` and
`schemas.DERIVED_TIMEFRAMES`. The locked plan CUT it (sec 5.2) with a stated
reason - no consumer - and the Phase 0.12 B3 higher-timeframe LRSI entry study
is one. Nothing is re-litigated: the cut's own condition is met, and the change
is purely additive. No existing timeframe, aggregation contract id, partition
layout or published row changes.

RTH is 6.5 hours, so two-hour buckets do not divide the session. H2 yields
three full buckets and a 30-minute STUB (15:30-16:00), exactly as H1 yields six
full hours and a stub, and H4 two buckets one of which is a 150-minute stub.
The stub machinery already existed and is reused unchanged: the stub keeps its
true duration and carries `is_stub`.

**The study excludes stubs from its oscillator input.** That is a B3 decision
rather than an aggregation one - the bar is still published, because a bar the
warehouse can build is evidence - but an EMA fed a 30-minute bar inside an H2
series would be measuring a duration that changes with the time of day, which
is not what "completed bars only" means.

**Reopen/close trigger.** If the B3 study is retired without a successor, H2
loses its consumer again and becomes a candidate for re-cutting. It is not
removed automatically: published `bar_derived` rows under `timeframe=H2` are
evidence and a partition is never deleted to tidy a constant.

**Where.** `research_warehouse/aggregate.py`, `research_warehouse/schemas.py`,
`tests/test_warehouse_aggregate.py`.

## BD-79 — The HTF LRSI study's short legs are unmirrored, and that is a feature choice

**Date:** 2026-09-01 (Phase 0.12 B2/B3).

**Decision.** The 16-recipe higher-timeframe LRSI grid reads ONE series -
`indicators/efficiency_lrsi` over the derived bar closes - for every leg. Longs
take `cross_up` through 50 and 20; shorts take `cross_down` through 50 and 80.
The mirrored-close idiom the live M5 engines use for their short detectors is
NOT used here.

**Why this is a decision and not a detail.** The formula clamps its numerator
at zero, so the series is not symmetric: a perfectly efficient DOWN move and a
motionless one both read 0. There are therefore two genuinely different short
features available, and they answer different questions - the mirrored one
measures how efficient the down move is, the unmirrored down-cross measures the
UP move's efficiency collapsing. `tests/fixtures/efficiency_lrsi_research_v1.json`
holds one series where the unmirrored down-cross fires at bar 27 and the
mirrored up-cross at bar 29, so the gap is a number rather than an argument.

Three reasons for the unmirrored reading, in order of weight:

1. **The four legs have to be comparable.** A grid whose long legs read one
   series and whose short legs read another is two studies sharing a table,
   and "does this line pay on the short side?" stops being answerable from it.
2. **The mirrored idiom is already the live engines'.** Re-running it in a
   shadow lane would produce a result that reads like evidence about a champion
   detector when it is not - the exact confusion plan.md sec 7 exists to stop.
3. **It matches how the trader reads the oscillator** - "the efficiency is
   going" - in the same units as the long legs.

**The cost, stated.** This measures EXHAUSTION, not down-momentum, and it fires
earlier. If the study later wants down-momentum, that is a SECOND registered
feature with its own recipes, never a reinterpretation of these rows.

**Live behaviour is untouched.** `CROSS_LEVELS` stays `(20.0, 50.0)`;
`RESEARCH_CROSS_LEVELS` is additive and read only by the shadow lane. No
`m5_signal_engines` behaviour changed.

**Where.** `indicators/efficiency_lrsi.py::RESEARCH_CROSS_LEVELS`;
`research_warehouse/outcomes.py::HTF_LRSI_RECIPES`,
`::simulate_htf_lrsi_entry`; `tests/test_warehouse_htf_lrsi.py`;
`tests/fixtures/efficiency_lrsi_research_v1.json`.

## BD-80 — The HTF LRSI rows are not registered in `outcome_semantics`

**Date:** 2026-09-01 (Phase 0.12 B4).

**Decision.** No family is added to `outcome_semantics.FAMILY_SPECS` for this
study, including the `lrsi_cross_80` its docstring names as a hypothetical.

**Why.** `outcome_semantics` classifies TRACKER/bounce families - the `family`
an outcome row carries out of `_make_bounce_event_id` - and decides what may be
averaged as a trade. The B3 rows are warehouse `outcome_path` rows keyed by
`recipe_id`; they never acquire a family and never reach `claim_kind`. The
existing M5-close recipe grid is registered nowhere either, for the same
reason. Registering `lrsi_cross_80` now would assert a claim kind for a family
with no producer anywhere in the tree, which is a statement this registry is
specifically built not to make on a guess.

**Reopen trigger.** If a live M5 engine ever emits an `lrsi_cross_80` family -
which would require a separate authorization, since the live `CROSS_LEVELS` are
unchanged - it must be registered in the same commit, or it reads
`unconfigured` and its rows are excluded from every mean.

**Where.** `outcome_semantics.py` (unchanged, deliberately);
`tests/test_warehouse_htf_lrsi.py::test_no_recipe_here_can_reach_a_champion`.
## BD-81 — Episodes are published beside rows; the floor still counts rows, and the follow-up is a CROSS-CELL floor

**Date:** 2026-09-01 (Phase 0.13 packet P3, item 1).

**Decision.** Every fact-pack cell now reports `n_episodes` — distinct
`dependency_cluster_id` — beside `n`. The eligibility floor **still counts
outcome rows**. The pack additionally publishes `evidence_shape`: rows, distinct
occurrences, distinct episodes and rows-per-occurrence over its whole
trade-family base.

**Why.** The ERD cardinality table is explicit that `setup_occurrence` →
`outcome_path` is 1:N and that "alternative recipes/horizons are correlated
diagnostics of ONE episode; they are never summed as independent samples". The
pack reported `n` as if rows were samples.

**What the measurement changed.** The first assumption was that a per-cell
episode count would be smaller than `n` and that the floor should simply move
onto it. Measured on the live lake 2026-09-01: 9,372 outcome rows rest on 599
occurrences and 287 clusters — but **inside a single (family, side, recipe)
cell, `n` and `n_episodes` were EQUAL in all 756 cells**. One row per occurrence
per recipe, so the per-cell count is not where the double-counting lives; 1,804
of 3,436 clusters carry more than one family, and there are 15.6 recipe rows per
occurrence.

The correlation is therefore **across cells**, not within one: nine ATR variants
of one family are nine readings of the same 33 moves. `evidence_shape` exists
because that is the denominator a reader comparing cells actually needs.

**Reopen trigger / follow-up.** Moving the floor changes which cells the model
may narrate, so it is its own packet. When it happens it must be a **cross-cell**
rule — an episode budget over the family, or a correlated-cell cap — and not the
per-cell swap first assumed, which on today's data would change nothing at all.
Publishing both counts now is what makes that packet decidable.

**Where.** `ai_jobs/setup_research.py::_summarize` (`n_episodes`,
`eligibility_rule`), `build_fact_pack` (`evidence_shape`);
`tests/test_setup_research_fact_pack_truth.py`.

## BD-82 — The fact pack leads with what cleared the floor, in two blocks

**Date:** 2026-09-01 (packet P3, item 2).

**Decision.** `policies` is built as two blocks: the **eligible** cells, whole
and sorted by trimmed mean as before, then a **bounded ineligible** block
(`MAX_INELIGIBLE_POLICY_ROWS = 40`) sorted by `n` DESC then trimmed mean. Drops
are counted per block (`eligible_policy_cells_dropped`,
`ineligible_policy_cells_dropped`); `policy_cells_dropped_from_pack` keeps its
meaning. The Markdown opens with the eligible block under its own heading.

**Why.** The 2026-08-31 pack sorted everything by trimmed mean into one list.
Its nine eligible cells were all `AVWAPE_TO_FIRST_DEV`/LONG against ATR stop
controls and all NEGATIVE; rows 10 onward were n=1 cells reading +2.9R, and the
80-row cap then dropped 508 more without saying which kind. A reader skimming
the top of that file learned the opposite of what the evidence said.

Ordering the ineligible block by `n` first — the shape the context-cell path
already used — means what rides along is the thickest evidence that has not
cleared the floor, never the luckiest single trade.

**Where.** `ai_jobs/setup_research.py::build_fact_pack`, `render_markdown`. A
pack published before the split still renders as its author published it: a pack
is never edited, and a new reading is a superseding sibling.

## BD-83 — Non-trade families are excluded by an explicit role map, and reported

**Date:** 2026-09-01 (packet P3, item 3).

**Decision.** `NON_TRADE_FAMILY_ROLES` names `GENERAL` = FALLBACK and
`FAVORITE_ZONE_WATCH` = WATCH_STATE. Those families are excluded from every
policy and context cell and published in a `non_trade_families` block with their
outcome-row, episode and occurrence counts. Everything not named is a TRADE
setup.

**Why.** Appendix C is already normative: General/Untagged is a "Diagnostic
fallback" that "must not become a pooled 'setup' edge", and Favorite Zone Watch
is a "Watch state" that is "never counted as a triggered trade setup". Nothing
in the nightly job knew the roles, so it pooled both — on the 2026-08-31 pack
that was 735 and 486 occurrences, and on today's lake 1,182 and 804 outcome
rows.

Counts still travel because **absence is a first-class fact**: a family that
simply is not in the table reads as one with nothing to say.

**Default TRADE, deliberately.** A family added tomorrow is measured rather than
silently excluded, and excluding a real setup takes someone typing its name.

**Reopen trigger.** Packet P7's setup registry owns `role` as a column
(`TRADE_SETUP`, `CONTEXT`, `WATCH_STATE`, `CONTROL`, `FALLBACK`). When it lands,
this constant goes and the role is read from the registry row.

**Where.** `ai_jobs/setup_research.py` (`NON_TRADE_FAMILY_ROLES`, `family_role`,
`build_fact_pack`, `render_markdown`).

## BD-84 — Outcome bucket coverage is recorded per firing, under the store root

**Date:** 2026-09-01 (packet P3, item 4).

**Decision.** New `research_warehouse/outcome_coverage.py`: an append-only JSONL
under `<store root>/_diagnostics/`, one line per outcome firing naming the
symbol bucket it covered. `run_build` appends after the outcomes step;
`ai_jobs/setup_research` reads the last 32 firings into the pack's `coverage`
alongside families with zero outcome rows and the first M5 session in the lake.

**Why.** `cli._run_outcomes` simulates ONE of 32 symbol buckets per firing, so a
family can be missing from a pack for two opposite reasons: it was measured and
produced nothing, or its symbols have not come up yet. The pack could not tell
them apart. "Not measured yet" must read differently from "measured and flat".

**No history reads UNKNOWN, never "0 of 32"** — a zero there is a measured claim
nobody measured. A step that never reached a bucket (`NO_OCCURRENCES`, a refused
lock) is not recorded rather than logged as covering bucket 0. A failed append
returns False and logs at debug: the outcome rows are the product, and this is
evidence about them.

**Location, and a deviation from the packet as written.** The packet asked for
the sidecar "beside the packs", i.e. in the AI store. That would make
`research_warehouse.cli` — the data layer — import `ai_jobs.store`, inverting the
one-way dependency the tree keeps (`ai_jobs` reads `research_warehouse`, never
the reverse). It lives under the store root instead, beside the lake it
describes. The reader already imports this package, so the pack still gets the
number.

**`first_m5_session` reads partition NAMES from the manifest**, never bar rows:
materialising a month of M5 bars to learn its own name is precisely the mistake
the month-keyed read rules exist to prevent (BD-66/BD-69).

**Where.** `research_warehouse/outcome_coverage.py`; `cli.py::run_build`;
`ai_jobs/setup_research.py::_coverage_state`, `_first_m5_session`,
`_coverage_lines`.

## BD-85 — `slice_readout` can read every family; `SLICE_SETUPS` is not widened

**Date:** 2026-09-01 (packet P3, item 5).

**Decision.** `slice_readout` gains `setups`. Omitted (a sentinel, not `None`)
means `SLICE_SETUPS`, so every existing caller is byte-identical; `None` means
every family present in the lake; a collection means exactly those. The Research
readout panel gains a family combo defaulting to the pinned slice, plus the four
columns the query always computed and the panel dropped — `n_symbols`,
`n_sessions`, `n_truncated`, `as_observed_only`.

**Why `SLICE_SETUPS` stays two.** It is the pinned Phase-6 vertical slice AND
`cli._run_outcomes` uses it to choose which occurrences get the legacy slice
recipe. Widening it would change what the warehouse **simulates**, not just what
a reader is shown. The argument changes only the reading.

**Why the panel needed it.** The occurrence table holds far more than the slice,
and a panel that can only ever show two families cannot answer "is anything else
being measured?" — the question the coverage work above exists to make askable.

**Selecting a family reads nothing.** Refresh remains the only thing that
touches the share (plan sec 20), still off-thread, and the EXPLORATORY caveat
now explains what the new columns mean.

**Where.** `research_warehouse/queries.py::slice_readout`;
`ui/panels/warehouse_readout_panel.py`.

> Numbering note: BD-78 and BD-79 are taken by the Phase 0.12 higher-timeframe
> LRSI work on `claude/focus-declutter-lrsi-htf`, which is unmerged. P3 branched
> off `main` and continues from BD-81 so the two lines cannot collide.
## BD-86 — The setup crosswalk is frozen data, and it resolves nothing it was not told

**Date:** 2026-09-01 (Phase 0.13 packet P7).

**Numbering note, updated at the 2026-09-02 merge.** These two were written as
BD-85/86 on the branch, when BD-78..84 were claimed on unmerged branches and
BD-80 was double-claimed by `claude/focus-declutter-lrsi-htf` (78-80) and
`claude/p3-fact-pack-truth` (80-84). All three merged to `main` on 2026-09-02 in
that order: the LRSI branch kept BD-80, P3's five shifted to 81-85, and these two
shifted again to 86/87. The file's headings are now 77..87 with no repeats, and
the merge asserted that. **The lesson is cheap to state and was expensive to
resolve: a BD number claimed on a branch is a request, not a number.**

**Decision.** `scripts/setup_registry.py` loads a FROZEN
`setup_registry_v1.json` (57 entries, keyed `setup_id@version`) generated by
`scripts/build_setup_registry.py`. It is never rebuilt at import. Where two
naming sites disagree about one setup, the registry records the disagreement in
`known_divergences` and resolves nothing; where Appendix C requires a column no
source establishes - supported sides, timeframe roles, the exact completed-bar
trigger, the primary recipe - the row is EMPTY and lists the field under
`unestablished`. An unresolvable name raises rather than defaulting.

**Why.** A crosswalk that recomputes itself from five moving sources is a sixth
source: its disagreements appear and vanish without anyone seeing them, and the
one thing a crosswalk exists to make visible is exactly that. Freezing turns a
divergence into a reviewable diff.

The two refusals are the same rule twice. Choosing which of two spellings is
IDENTITY is a decision (plan.md P4.1 owns it), not something a generator may
derive from whichever table it happened to read first. And a guessed
`supported_sides` reads as established in exactly the column a later experiment
trusts - the blank says "not established", which is true, and the guess says
something false in a machine-readable field.

**The packet named four sources; the code has five.** `legacy.py` declares study
families as `*_STUDY_FAMILY` constants and eight of them are named nowhere else.
They are read by regex - no import of a 27k-line module, no write to it - and a
test pins the count so a moved constant fails loudly instead of silently
shrinking the registry.

**Where.** `scripts/setup_registry.py`, `scripts/build_setup_registry.py`,
`scripts/setup_registry_v1.json`, `tests/test_setup_registry_and_trial_ledger.py`,
`packaging/tradingbotv3.spec` (the scripts-root asset sweep the frozen JSON
required).

## BD-87 — A trial is declared before the numbers, and never rewritten after them

**Date:** 2026-09-01 (Phase 0.13 packet P7).

**Decision.** `scripts/research_warehouse/trial_ledger.py` writes one
append-only JSONL row per registered grid at
`<store root>/_diagnostics/trial_ledger.jsonl`: question, failure mode, declared
cells, declared cell count, declared floors, declared window, authorization
pointer, status. `register` REFUSES a `trial_id` the ledger already carries, and
nothing in the module can read an outcome (a test bans `read_rows`,
`latest_outcomes`, `ResearchStore` and `total_r` from its body). The four grids
that predate the ledger are backfilled with their real authorization pointers:
the M5-close recipe grid (54 cells), the HTF LRSI entry grid (16), the AVWAP band
challenger (3) and the v1 recipe library (5).

**Why.** Section 15.1 replaces formal multiplicity machinery with a count -
`n_variants_examined`, family-lifetime, never reset by splitting the search
across files - and the widening rule (k>10 implies a 99% holdout interval AND
beating the family median) is not computable unless the grid's size was written
down before anyone looked. The refusal to rewrite is the whole mechanism: a
declaration that can be edited after the numbers arrive is how a grid of 54 cells
becomes a grid of 3 in the record. A genuine change of plan is a NEW trial id.

Backfilling rather than starting empty is the same argument: a lifetime count
that began the day the ledger shipped would report every one of these families as
never having been examined, which is the specific understatement k exists to
prevent.

**Membership is explicit-then-prefix.** A row claims recipes by an explicit
`recipe_ids` list first and its `recipe_id_prefix` second, so a named recipe can
never be captured by another row's prefix. `owners_of` returns EVERY claimant
rather than the first, because the interesting failure is not "no owner" but
"two owners" - one result counted against two families' look counts.

**Where.** `scripts/research_warehouse/trial_ledger.py`,
`tests/test_setup_registry_and_trial_ledger.py`.

## BD-88 — A parameter grid varies ONE factor, and its control is the code it challenges

**Date:** 2026-09-02 (Phase 0.13 packet P8).

**Decision.** The entry-timing grid holds the stop (`current_anchor:1`), the
time stop, the exit machine and the checkpoints identical across all twelve
cells; only the entry moment and the target vary, and the target is the axis the
grid is compared ALONG rather than a second experiment. The control cells
(`m5_first_close`) do not reimplement the next-session entry - they call the
existing `simulate_m5_close_opportunity` with the existing rank-1 selector, so
they reproduce the `m5close_current_anchor1_*` rows by construction. The three
challengers call the SAME function through one new optional `entry_selector`.

**Why.** A grid that varied the stop as well could not answer the question it
declared: a cell that won might have won on the stop, and nothing in the row
would say which. And a control that is a separate implementation is not a
control - two copies of an exit loop eventually disagree, and the disagreement
would present itself as a finding about entries. Delegation makes parity a
property of the code rather than a claim a test has to keep re-checking; the
test then pins it anyway, because "by construction" is only true until someone
edits one of the two paths.

**The golden fixture was pinned from code that had never heard of P8.**
`build_setup_entry_timing_fixture.py` imports `outcomes.py` as `main` has it
(through `git show`, into a temp package) and freezes the three rank-1 rows from
THAT. The packet asked for a fixture before the simulator; the arithmetic that
actually needed protecting was the arithmetic that already ships, since P8 adds a
parameter to a function every published `m5close_*` row came from.

**Reopen trigger.** If a later packet needs to vary the stop inside this family,
it is a NEW grid with its own trial-ledger row and its own k, not a widening of
this one.

**Correction, review round R1 (2026-09-02).** This entry originally said the
derived series were memoised per occurrence. **They were not** -
`_entry_from_derived` called `_htf_series` fresh on every cell, so one
occurrence rebuilt the same M15 series three times and the same M30 series three
more. Measured: **2.06 s per occurrence, of which ~0.8 s was rebuilding series
already built.** The cache `simulate_htf_lrsi_entry` uses is now threaded
through the same way - handed in by the caller, keyed by symbol/timeframe/cutoff,
dropped with the occurrence, never module-level. A test counts the `_htf_series`
calls and fails if the M15 series is built more than once for one occurrence.

The parity fixture's baseline was also a moving target: it read `main`, which is
correct until P8 merges and then becomes a self-portrait - a rerun would compare
the new code against itself. It is pinned to `1837b63`, the last commit that had
never heard of this grid.

**Where.** `research_warehouse/outcomes.py` (`SETUP_ENTRY_TIMING_*`,
`simulate_setup_entry_timing`, the `entry_selector` parameter),
`scripts/build_setup_entry_timing_fixture.py`,
`tests/fixtures/setup_entry_timing_parity_v1.json`,
`tests/test_setup_entry_timing_grid.py`.

## BD-89 — A confirmation entry is defined by what it REFUSES, and its denominator is the control's

**Date:** 2026-09-02 (Phase 0.13 packet P8).

**Decision.** Each of the three confirmation entries is spelled out rather than
implied. Acceptance is a completed M15 CLOSE beyond the trigger, not a wick
through it. A retest is a completed M5 bar that TAGS the trigger level and still
CLOSES holding it. A controlled pullback is a completed M30 bar with the EMAs in
trend order, an extreme reaching the band and a close still beyond it - a bar
that closes THROUGH the band is a break. All three read the warehouse's own
derived bars with stubs excluded, and all three are eligible only STRICTLY after
the occurrence's trigger. `_ema_series` returns `None` until its window is full,
so a family with fewer than 21 completed M30 bars produces NO ROW.

**Why the strict-after rule.** A derived bar whose interval ENDS at the trigger
instant is the signal bar itself. Entering on it would be entering on the
information that created the setup, which is the look-ahead this whole module
exists to prevent - and it is a rule the M5 control already had
(`interval_end > trigger`), so the derived path matching it is consistency, not
caution.

**THE FAILURE MODE THIS GRID IS MOST LIKELY TO PRODUCE, recorded before any
number exists.** A waiting entry can look better purely because it SKIPS the
episodes that went straight down: the confirmation never printed, so no row was
written, and the loss is missing from the average rather than counted. The
control's rows-per-cluster is therefore the denominator to read FIRST, and a
challenger with materially fewer rows is reporting survivorship rather than edge.
The second failure mode is the three challengers agreeing so strongly that they
are one look and not three. Both are in the trial-ledger row, written at
registration, because a failure mode named after the numbers arrive is a
rationalisation.

**Where.** `research_warehouse/outcomes.py` (`_entry_from_derived`,
`_entry_after_m15_acceptance`, `_entry_after_m5_retest`,
`_entry_after_m30_ema_pullback`, `_ema_series`);
`research_warehouse/trial_ledger.py` (the registered row).


## BD-90 — A like links to an occurrence by a stated BASIS, and absence is a row

**Decision (P10 B2, 2026-09-02).** `like_links` writes one row per like: the same
family the click recorded (`exact_family`), else any family by nearest trigger
(`any_family`), else `none`. The window is **one session back and five forward**.

**Why.** Nothing in the tree joined a like to a warehouse occurrence — the
round-1 audit's item 6, still unbuilt when P10 landed. Without it a like is a
symbol and a timestamp: what the setup looked like cannot be asked, and what a
different entry would have done cannot be simulated.

The asymmetric window is the trader's own: *"if I like a stock one day it may not
be for 3-5 days later that the best entry is."* Back one session because a like
is usually made on a setup that has already triggered and the trader may be
reading the previous close.

**A like with no occurrence is written with basis `none`.** A study that dropped
the unmatched likes would report on the subset the scanner happened to find,
which is precisely the population whose behaviour differs. `candidates_in_window`
rides along because `any_family` with eleven candidates is a much weaker claim
than with one, and the count is the only thing that says so. A like the join
cannot ADDRESS — no side, no symbol, no id — is skipped rather than written as
`none`, so a phantom never enters the denominator.

**Bronze, not a new gold schema.** The slice datasets are frozen (plan sec 7.1)
and the bronze namespace exists so an additive artifact needs no schema change:
`bronze_like_occurrence_link` on the shared `BRONZE_RECORD`, link fields in the
JSON payload, record hash over that payload so a re-run is idempotent.

**Reopen if** the `none` share stays high after several weeks — that would mean
the window or the side rule is wrong, not that the trader likes unfindable names.

## BD-91 — `read_rows` can name its time column; the predicate stays a comparison

**Decision (P10, 2026-09-02).** `ResearchStore.read_rows` gains
`time_column="interval_start"`. The default is what every caller before P10 used,
so all of them are unchanged.

**Why.** The bar datasets are not the only ones with a time worth narrowing on.
`setup_occurrence` carries `trigger_at` and `event_at` and no `interval_start` at
all, so a caller wanting a date window there had to pull the partition into
Python and filter it — the exact cost BD-74 exists to prevent.

It is a NAME, never an expression: the predicate built from it is the same
half-open `[start, end)` comparison, so a caller cannot express a filter that
silently means something other than the Python one it replaces.

## BD-92 — An after-like outcome row is keyed by the LIKE EPISODE, not the occurrence

**Decision (P10 C2, 2026-09-02).** After-like rows carry
`occurrence_id = afterlike|SYMBOL|SIDE|like_date` — the dependency cluster —
rather than the linked occurrence's own id.

**Why.** `outcome_path`'s grain is `(occurrence_id, recipe_id,
outcome_definition_id)`. Two likes on two days that link to the same occurrence
produce genuinely different rows, because each offset is measured from its own
like's session; under the occurrence's id they collide on that grain and the
second silently replaces the first.

It also keeps an after-like row from being mistaken for an occurrence outcome by
any join on `occurrence_id`, and it makes the episode count free: distinct
`occurrence_id` values ARE distinct likes, which is what the declared floors
count.

**Consequence, stated rather than hidden:** `outcome_path` has no column for the
linked occurrence, so the family behind an after-like row is not recoverable from
the row alone. The fact pack's `after_like` block says so in a `family_split`
field instead of quietly omitting the split. The join exists — through
`bronze_like_occurrence_link` — and is a reader's step.

## BD-93 — The unlinked bucket is a COUNT, because the declared stop needs an anchor

**Decision (P10 C2, 2026-09-02).** The after-like grid grades LINKED likes only.
Unlinked ones are counted by named reason and reported beside the cells.

**Why.** The registered grid declares one structural stop, `current_anchor:1`,
and that level comes from the occurrence's own tracker geometry. A like the
scanner never found has no anchor, so there is nothing to place the stop at.

Both alternatives were worse. A substitute stop for the unlinked bucket would
mean the grid no longer has ONE stop model, so an unlinked-vs-linked difference
could be a difference in stops rather than in the names. Dropping them silently
would hide how many of the trader's likes the scanner never found — which is
itself one of the more interesting things this study can report.

**Reopen if** a stop model that needs no tracker geometry is registered for this
grid — as its own trial, not as a patch to this one.


## BD-94 — A bronze link row is partitioned by the LIKE, not by the run

**Decision (R4 fix round 1, 2026-09-02).** `like_links.link_rows_for_bronze` sets
`event_at` and `partition_ts` from the like's OWN date
(`like_links.like_event_at`), not from the nightly run stamp. `observed_at` still
carries the run stamp, which is what the point-in-time contract says it means -
when this installation received the row.

**Why.** `bronze_like_occurrence_link` is month-partitioned on `partition_ts`,
the nightly after-like pass looks back 30 days, and the caller's dedup reads the
row's OWN partition because BD-74 forbids a month-wide read. With the run stamp
in `partition_ts`, a like from late September was written into September's
partition on the 26th and into OCTOBER'S on the 1st, and the dedup could not see
the earlier copy. Reproduced in a temp lake with one like dated 2026-09-25 over
passes on 09-26 / 10-01 / 10-02: the same `record_hash` landed twice. Every count
over the dataset, and the BD-92 join that is the only route from an after-like
outcome row back to its setup family, over-counted by however many month
boundaries the lookback had crossed.

The alternative was to widen the dedup to read every partition, which is the
month-keyed full read BD-74 exists to prevent and which grows for as long as the
trader keeps liking things.

**The frozen schema is untouched** (plan sec 7.1): same dataset, same columns,
same `BRONZE_RECORD`, same payload, same `record_hash`. Only which month a row
lands in moved - and it moved to the one the row is ABOUT, which is what
`event_at` was always specified to carry.

**A like with no readable date falls back to the run stamp.** A row filed under
the wrong month beats a row that cannot be written, and the fallback is bounded
to likes whose own date is missing or unparseable.

**Reopen if** the like log ever gains a real intraday timestamp worth keeping in
`event_at`; the partition would be unaffected (still that month) and only the
hour would sharpen.


## BD-95 — The post-scan build runs in a CHILD PROCESS, and the calendar is memoized

**Decision (packet F1, 2026-09-03, trader-authorized: "the program has been
freezing and has been basically unusable all morning ... fix it").** The post-scan
warehouse build leaves the desk process. `ScanService.start_warehouse_build`
spawns `research_warehouse.cli build --run-id <id>` with
`subprocess.Popen(..., creationflags=BELOW_NORMAL_PRIORITY_CLASS |
CREATE_NO_WINDOW)` — both flags read by name through `getattr` so macOS still
launches — registers it with `_register_owned_process`, and waits on it from one
daemon thread that blocks on the child's pipe. `_run_warehouse_build` is deleted.
Separately, `exchange_calendar.holidays`, `.half_days` and the session builder
behind `trading_session` are `functools.lru_cache`d.

**Why, measured on the desk (pid 11612, 2026-09-03 08:45–08:55 PT, `uvx py-spy
record --gil`).** The `qt-warehouse-build` thread held the GIL in **82.7%** of
samples; `MainThread` got **2.3%**. WM_NULL pings to the desk window from outside
the process measured 100–606 ms hangs every few seconds — that is the freeze the
trader reported. **84%** of that thread's samples were inside
`exchange_calendar.py` (`session_for` → `trading_session` → `is_trading_day` →
`holidays(year)`), recomputing Easter and five nth-weekday walks once per M5 bar
per occurrence. `manifest_log.jsonl` shows the `m5_close_recipe_outcomes` stage
taking **27–57 minutes** after every scan (09-01: 28/51/57; 09-02: 27/38/44), four
scans a day, all inside RTH.

Memoizing the calendar is a 21× speed-up on its own (20,000 `session_for` calls:
0.25 s → 0.0114 s in the desk venv) but it does not make the build small enough to
belong on a thread: a CPU-bound Python thread holds the GIL by construction, and
no priority, timer or chunking trick gives the GUI thread back. LD-01 specified
this work as a *post-scan/EOD CLI build job*; running it in-process was the
deviation, and it is reverted.

**What did not change.** The build itself, its single-flight lock, its manifest
authority, the parent-side `warehouse_enabled()` gate (no store → no child), the
one-build-at-a-time rule, and `wait_for_warehouse_build`'s name and role on the
shutdown path. A reaped child is safe because `single_flight` reclaims a dead
holder's lock rather than obeying it. The lru_cache sits behind `trading_session`
in a positional `_trading_session(day, calendar)` because `lru_cache` keys on the
call SHAPE, and every in-module caller passes `calendar=` as a keyword — decorated
directly, the same day would be built and stored twice. `TradingSession` is a
frozen dataclass, so one shared instance is safe; the holiday dicts are shared and
**no caller may mutate them** (every caller in `scripts/` and `tests/` only reads,
checked 2026-09-03).

**The build child is OWNED without being a SCAN child, and that distinction is
load-bearing.** `ScanService._start` refuses a new scan while
`owned_scan_process_count()` is non-zero ("previous scan child still running"),
so counting the build there would have turned a freeze fix into skipped scans —
a 27–57 minute build, four times a session, straddling the next slot every time.
The child is registered for the shutdown reap and for
`owned_scan_process_snapshot` (which is about reaping and answers plan sec 6.1's
"owned child-process count returns to zero"), and it is tracked in a second list
so `owned_scan_process_count` keeps its exact previous meaning.
`owned_build_process_count()` answers for builds. The second list holds process
OBJECTS rather than pids, because the OS recycles pids.

**Reopen if** the build ever needs to hand a live object back to the desk (it does
not — it publishes through the lake), or if a platform without
`BELOW_NORMAL_PRIORITY_CLASS` needs a real nice level rather than the 0 the
`getattr` falls back to.

## BD-96 — The tee de-duplicates BEFORE it works, the mark is persisted, and the seal de-duplicates too

**Decision (2026-09-03 evening, trader-authorized: "go ahead and implement all
packets" on the desk assessment).** Three changes, one defect.

1. `bar_archive.capture_m5_tee` now runs in two passes. Pass 1 parses only the
   bar's timestamp, drops forming bars and anything already captured, and
   stages the rest; pass 2 parses prices, hashes and session-tags only the bars
   that will actually be published. The session lookup is cached per session
   DATE inside the call, and `_market_session_module` is memoized so the
   `Path(__file__).resolve()` it used to run on every call runs once per
   process. A new optional `high_water={symbol: newest interval_start}` is the
   live desk's dedupe state: a bar at or before its symbol's mark is a
   duplicate, and a symbol whose NEWEST bar is at or before the mark is counted
   as `symbols_unchanged` and never walked - the champion's cache is sorted by
   `_dedupe_bars`, so the last bar answers for the list.
2. `WarehouseTeeCapture` keeps that mark instead of a `seen` set, and persists
   it as `tee_high_water.json` in the spool directory (atomic replace, 14-day
   retention per symbol, unreadable file = empty mark and a warning, never a
   raise). It is never reset by the clock.
3. `spool.seal_spool` de-duplicates at the dataset grain: rows whose grain key
   is already live in the target partition, or already sealed earlier in the
   same call, are dropped and COUNTED (`SealResult.rows_deduplicated`).
   `SUPERSEDING_DATASETS` are exempt. The lake read is grain columns only, per
   touched partition, in the build child.

Repair: `ResearchStore.duplicate_rows` (dry run) and `dedupe_partition`, and the
CLI `research_warehouse.cli dedupe [--dataset] [--partition ...] [--apply]`
(dry run without `--apply`). `dedupe_partition` is a COMPACT-shaped rewrite -
one manifest line, inputs retired never deleted, restorable by repointing the
manifest - that is allowed to make the partition SMALLER and says by how much
(`rows_dropped`, `dedupe_grain` on the line). It keeps the row with the earliest
`observed_at`: the first capture is the evidence, a later twin is the tee
re-offering it.

**Why, measured on the desk (pid 18548, main `f903ca4`, 2026-09-03 21:02-21:10
PT, after the close).** The process had used 29,909 CPU-seconds in eight hours;
thread `warehouse-m5-tee` alone had 26,540 of them and was at 101% of one core
at 21:05. `uvx py-spy record --gil` for 15 s: 330 of 362 samples (91%) in
`capture_m5_tee`, the largest leaf `session_context` -> `_market_session_module`
-> `_ensure_scripts_on_path` -> `Path.resolve()` (197 us each, benchmarked), then
`get_market_session_window`, then `_source_hash`. The 60-second timer handed the
tee the whole `latest_bars` cache every tick - 888 symbols x 5 sessions x 78 bars
= 346,111 bars after the close - and every one of them was parsed, session-tagged
and hashed BEFORE the `seen` check dropped it. That is >=72 s of work per walk
against a 60 s timer; the one-slot mailbox meant the thread never queued and
never rested. The GUI thread appeared in 0 of 362 GIL samples.

The second half of the defect: `_session_seen` keyed the `seen` set on
`moment.date()` of a UTC moment, so at 00:00 UTC (17:00 PT) the set emptied and
the tee re-spooled the whole cache - `segment-20260904T000029-*.open.jsonl`
holds 346,111 rows / 240 MB for five sessions, four of them already in the lake -
and a restart did the same (107,119 rows at 13:05 PT). `seal_spool` published
whatever the spool held ("the tee de-duplicates before spooling"), so the lake
now carries the duplicates: `bar_m5 month=2026-08` 12,015,283 rows for 1,816,970
distinct grain keys (10,198,313 duplicates, 85%); `month=2026-09` 541,444 rows
for 208,841 keys (332,603 duplicates). `bar_d1`, `bar_derived` and
`feature_snapshot_intraday` carry NO grain duplicates (checked with the dry run),
but the derived and feature rows for those months were COMPUTED FROM the
duplicated M5 rows: `aggregate_symbol_session` counts every twin as a
constituent (volume x N, `constituent_count` > expected, quality PARTIAL) and
`compute_intraday_features` windows over the doubled list. Those rows need a
rebuild after the dedupe; that rebuild is an overnight job and is recorded as
owed, not run here.

**What did not change.** The tee still issues no provider request and touches no
lake; `seen=` still works for the CLI and backfill callers; the seal's
`sealed_before` crash guard stands beside the new dedupe; `compact` is untouched
and still refuses a row-count change (a compaction that loses rows is a defect,
a dedupe that loses rows is the point, and the two are different verbs).

**Reopen if** the champion's cache ever stops being sorted (then the last-bar
short-circuit must go back to a full walk), or if a dataset other than the
superseding two legitimately needs a repeated grain key at the seal.

## BD-97 — A polluted derived partition is RETIRED whole and recomputed, never patched

**Decision (2026-09-03 late evening, trader-authorized: "implement the rest").**
`ResearchStore.retire_partition(dataset, partition)` appends ONE `RETIRE` manifest
line whose `supersedes` names every live file of the partition (no replacement
file, a `reason` on the line); the files leave the live set at once and move to
`_retired/<day>/` on the next GC, restorable by repointing the manifest. Refused
for non-compactable datasets (bronze raw / evidence). The CLI
`research_warehouse.cli rebuild-month --month YYYY-MM [--apply]` (dry run by
default; `--apply` under the build's single-flight lock) retires that month's
`bar_derived` (every timeframe) and `feature_snapshot_intraday` partitions and
then, for every exchange session of the month, runs the derived-bar, weekly-bar
and intraday-feature steps exactly as the nightly build does, with
`run_id="rebuild_month"` on every row.

**Why.** BD-96's duplicated `bar_m5` polluted its DERIVED datasets in VALUE, not
by duplication: `aggregate_symbol_session` counted every twin as a constituent
(`constituent_count` 6 where 3 were expected, volume x2, quality PARTIAL - pinned
by `tests/test_warehouse_rebuild_month.py`), and `compute_intraday_features`
windowed over the doubled list. Neither carries a repeated grain key, so
`dedupe` cannot repair them; and `build_intraday_snapshots` skips every
`ALREADY_COMPUTED` key, so recomputing without retiring first would change
nothing. Retire-then-rebuild is the only shape that both repairs and leaves the
audit trail: the polluted parts are still on disk for 30 days, and the RETIRE
line says why they left.

**Not rebuilt: outcomes.** `outcome_path` rows for those months were computed
over the doubled series (bar-count horizons stretched; price paths unchanged).
They are SUPERSEDING rows revisited by the nightly build's bucket rotation, and a
month-wide outcome recomputation is its own overnight job - recorded as owed in
plan.md Phase 0.15, not run.

**Runbook (trader's commands, in this order, with no build running):**

    cd scripts
    ..\.venv\Scripts\python.exe -m research_warehouse.cli dedupe --dataset bar_m5 --apply
    ..\.venv\Scripts\python.exe -m research_warehouse.cli rebuild-month --month 2026-08 --apply
    ..\.venv\Scripts\python.exe -m research_warehouse.cli rebuild-month --month 2026-09 --apply

**RUN 2026-09-03 22:29-22:42 PT by the lead with the trader's explicit permission**
(an earlier attempt was refused by the session's permission classifier): dedupe
dropped 10,198,313 + 332,603 rows (two COMPACT lines); August retired 250 files
across 5 partitions and recomputed 21 sessions (1,072,253 derived, 5,825 weekly,
1,816,970 intraday feature rows - one per repaired M5 bar); September retired 44
files across 4 partitions and recomputed 4 sessions (123,705 / 208,841). GC moved
483 + 44 retired files. Checked afterwards: M15 August has 15 over-counted rows of
605,909 (0.002%) and 99.9% COMPLETE; September 0 of 70,047. The whole repair took
~13 minutes under the build lock while the nightly AI-jobs runner was up.

**Reopen if** the feature step ever stops skipping already-computed keys (then the
retire becomes optional), or if outcomes gain a per-month recompute entry point.

## BD-98 — A forced outcome recompute exists, for inputs that turned out to be wrong

**Decision (2026-09-04, trader-authorized: "start these last 2 projects").**
`outcomes.build_outcomes` takes `force: bool = False`. With it, a terminal row
(`TERMINAL_RESULT_STATES`) is re-simulated like a non-terminal one; the
"idempotent by knowledge" rule otherwise stands. A re-simulation that reproduces
the stored result still writes nothing (`_same_outcome`); only a changed result
supersedes, so the current view is repaired in place and the old row stays in the
ledger. `_run_outcomes` takes an explicit `bucket` and passes `force` down to both
`build_outcomes` calls (the M5-close grid and the legacy slice).
`research_warehouse.cli recompute-outcomes [--buckets a-b,c] [--time-budget-minutes
N] [--session-date D] [--apply]` walks the buckets with force, taking the
single-flight lock ONCE PER BUCKET so a scheduled build slots in between, records
one coverage firing per bucket (`run_id=outcomes_recompute-bNN`), and stops
starting new buckets once the budget is spent - the report and the coverage file
name what a later run still owes. Dry run by default.

**Why.** BD-96/97 repaired `bar_m5` and its derived datasets, but the outcome rows
for 2026-08/09 were simulated over the doubled M5 series (bar-count horizons
stretched, price paths intact) and are mostly TERMINAL, which the nightly build
never touches by design. Without `force` they would have stayed wrong forever; with
a blanket recompute they would be rewritten even where nothing changed. The
combination - force plus write-only-on-change - is the narrowest repair.

**What did not change.** The nightly build never passes `force` or `bucket`; the
(day, hour) bucket rotation, the after-like pass and the market-bias context are
untouched; the supersession rule (`SUPERSEDING_DATASETS`, latest by
`computed_at`) is what makes the repair a normal write.

**Run.** Started 2026-09-04 07:00 PT against the live lake, 32 buckets over 6,850
occurrences / 1,715 symbols, 340-minute budget; bucket 0 (48 symbols, 189
occurrences) took ~2.5 minutes, so the walk runs alongside the session's scans,
each post-scan build refused only while a bucket is mid-flight. Finished
07:53 PT: 32/32 buckets in 53 minutes, 6,850 occurrences, 134,502 rows superseded
(changed result), 3,803 unchanged, 423,395 cells INSUFFICIENT_PATH_DATA, zero errors
or refusals - the desk's 07:30 scan and its build ran alongside without a collision.

**Reopen if** a recipe ever needs a per-row "inputs revision" so a stale row can
be detected instead of re-simulated wholesale.

## BD-99 — An anchor's KNOWLEDGE travels with the snapshot, and a reconstructed one is research evidence only

**Decision (2026-09-04, packet Q2.1).** `cli.anchor_dates_by_symbol(store, day)`
returns `{symbol: features.AnchorChoice}` — the anchor bar date it always
returned, plus a `knowledge` of `observed` or `reconstructed`. `observed` means
the row's own `system_from`, converted to market-local with `astimezone` (never
by stripping the timezone), lands **on or before** the session; anything later,
and anything with no stamp at all, is `reconstructed`. The
"an anchor that had not happened yet is not knowable" exclusion
(`anchor_bar_date > day`) is unchanged. Where one bar date carries several
rows the **earliest** `system_from` wins — the first moment it became knowable.
`feature_snapshot_daily` gains the additive column `anchor_knowledge`, written
only where an anchor was actually used (`""` when the row used none); a row
written before the column reads NULL and `features.anchor_knowledge_bucket`
calls it **`legacy`**, never `observed`.

**Why.** The 2026-09-04 earnings-anchor bridge appends ~2,200 anchors whose
BARS are months old and whose knowledge stamp is that night. Sec 6.2 already
says an anchor is available once observed and not retroactively at its bar, and
`anchor_dates_by_symbol` was ignoring `system_from` entirely — so a daily
snapshot rebuilt for 2026-08-12 would have presented tonight's imports as
something the desk knew that day. The band repair (BD-100) is worth doing and
the rows it produces are legitimate RESEARCH evidence; they are not
point-in-time evidence and must never be counted as such.

**The rule that binds.** A `reconstructed` row is evidence for research and
**never for a plan.md sec 7 promotion gate**: sec 7 requires "a declared
evidence window frozen before inspection", and a knowledge stamp from after the
session is the opposite of that. Every reader that pools these rows reports the
split; `band-coverage` (Q2.3) prints it by bucket.

**What did not change.** No number moves. The bands, the favourite-zone block
and the AVWAP formula are untouched; `build_daily_snapshots` still accepts a
bare date from any caller that has one and reads it as `reconstructed`, because
a caller that did not state the knowledge has not established it.

**Reopen if** anchors ever arrive with a trustworthy "known at" of their own
(a broker or vendor timestamp), in which case `observed` should be decided from
that field rather than from the lake's write time.

## BD-100 — Past daily features are rebuilt by their own command, and the gate #59 runbook is four steps

**Decision (2026-09-04, packet Q2.4).** `research_warehouse.cli
rebuild-daily-features --from YYYY-MM-DD --to YYYY-MM-DD [--apply]`, dry run by
default. It is a **sibling** of `rebuild-month`, not a third entry in
`REBUILD_DATASETS`, for one reason: `feature_snapshot_daily` is partitioned by
**year**, so "retire the month's partitions" — right for `bar_derived` and
`feature_snapshot_intraday`, which are month-keyed — would retire every other
month of that year with it. The mechanics are otherwise BD-97's: under the
build's single-flight lock, one RETIRE line per affected year partition (files
kept, restorable by repointing the manifest), then a **verbatim carry** of
every row outside the requested range, then a per-session
`features.build_daily_snapshots` with the Q2.1 stamped anchors. A second
`--apply` therefore supersedes rather than duplicating, and the carried rows
keep their own values, including a NULL `anchor_knowledge` where they predate
the column: the rebuild never relabels history. The carry is CHECKED, not
reported: a republish short by even one row, or with anything quarantined,
raises `LakeIntegrityError` naming the partition — that data was live a moment
earlier and its old file is already retired, so a count that cannot fail would
be no evidence at all. **If a session's `build_daily_snapshots` raises
mid-loop, the partition is already retired and the surviving rows are already
republished: nothing is destroyed (retired files stay on disk until GC and are
restorable by repointing the manifest), the sessions written so far stand, and
re-running the same range is idempotent — it retires what the partial run
wrote, carries the rest, and recomputes the whole range again.**

**What `observed` will actually look like.** The choice keeps the NEWEST anchor
bar on or before the session **regardless of knowledge** — that ordering is
deliberate and unchanged from before Q2 — so a bridged anchor whose bar is newer
DISPLACES a hand-imported one rather than losing to it. Once the bridge has run,
expect `reconstructed` to dominate and expect `observed` to be rare or zero; an
`observed` row appears only where a symbol's newest qualifying anchor bar was
already in the lake before that session. The label is there so a reader knows
which kind of evidence they are holding, and the coverage report's job is to
PRINT the split — not to reach a target in one bucket.

**Why.** The nightly build writes daily features for ONE day
(`day = session_date or stamp.date()`), so every August and early-September
session was computed before the anchors CSV held anything and carries null
AVWAP bands. `_bands_by_occurrence` correctly reads the TRIGGER session's own
feature row and correctly finds nothing, which is why `swing_house_v1` graded
0/257 and why `recompute-outcomes` alone could not have repaired it — there was
nothing new for it to read.

**The gate #59 runbook, in order.** 1. the nightly build (the bridge's anchors
are ingested into `anchor_instance`); 2. `rebuild-daily-features --from
2026-08-01 --to <today> --apply` (dry run first, read the session list);
3. `recompute-outcomes --apply` (BD-98 `force`, one lock per bucket);
4. `band-coverage --month 2026-08` and `--month 2026-09`. Step 3 before step 2
measures nothing new. Every step is dry-run-by-default, and none of them is run
against the live lake by an agent without the trader's go.

**Cost note.** The carry materialises the year partition's out-of-range rows
once. That is the price of a year-keyed partition; it is why this is a
maintenance command under the lock and never a step in the nightly build.

**Reopen if** the daily-feature partition is ever re-keyed by month, which
would let `rebuild-month` own it directly and delete the carry.

## BD-101 — The challenger's bands are a SECOND family beside the champion's, and its recipe is a twin

**Decision (2026-09-05, packet M4; trader: *"I want us to compare both to see
what is better"*, plan.md Phase 0.19 item 2, `docs/AVWAP_BAND_VARIANT_STUDY.md`
T3 step 4).**

**Nine additive columns, never a shared one.** `feature_snapshot_daily` gains
`avwap_variant_value`, `avwap_variant_stdev`, `avwap_variant_upper_1..3`,
`avwap_variant_lower_1..3` and `avwap_variant_formula_version`. The two families
never share a column and never will. `avwape_*` means *the champion's*
running-deviation sigma (decision 0008, frozen); `avwap_variant_*` means
*AVWAP(HLC/3) ± k · stdev(close, 20, population)*. One column that sometimes
meant either would make every stored row ambiguous the moment anybody changed a
default, and the whole point of the study is a PAIRED comparison — which needs
both numbers on one row, not one number and a flag.

**Computed independently of the champion, from the same bars and anchor index.**
`compute_daily_features` calls `features.avwap_variant_bands` → the pure
`indicators.avwap_band_variants.oneoption_avwap_bands`, and does so whether or
not the champion produced bands. The two formulas fail on DIFFERENT inputs: the
champion's sigma is zero by construction on a one-bar anchor (where OneOption
read 10.28), and the challenger's is `None` until twenty completed closes exist.
Gating the challenger on the champion's `if bands:` would have silently dropped a
measured band on exactly the rows the study is about.

**A NULL band is "not measured", and the formula version says which.**
`avwap_variant_formula_version` is written whenever the challenger was
ATTEMPTED — that is, whenever an anchor was used — so a row with a version and
NULL bands reads as "the sigma was unmeasurable here", while a row with neither
predates the column. The centre (`avwap_variant_value`) is reported even when the
sigma is not, because it IS measurable; what is never done is padding a band onto
the centre line.

**`FEATURE_SET_VERSION` → `tier1_v2`, and nothing is rewritten.** The dataset
identity is `(symbol, session_date, feature_set_version)`, so a `tier1_v1` row
and a `tier1_v2` row for one session are two rows and not a duplicate.
`rebuild-daily-features` (BD-100) supersedes; the nightly simply adds the new
shape. **Consequence, and it is the part that needed a fix**: a reader keyed on
`(symbol, session)` could now hold two candidates, and `_bands_by_occurrence` and
`run_band_coverage` took whichever landed last — file order. Both now keep the
newest `computed_at`, a tie keeping what is already held.

**The twin recipe is a `dataclasses.replace`, not a copy.**
`swing_house_variant_v1` is built FROM `SWING_HOUSE_V1`, so a later change to the
champion's management is inherited rather than forgotten: entry, stop model,
management, targets, expiry, `analysis_unit` and `required_band_numbers` are the
same object's values, and exactly three fields differ — `recipe_id`,
`band_family` and `outcome_definition_id`. That is what makes the comparison a
measurement of the BANDS rather than of two policies.

**The recipe names its family; the caller does not choose.** `Recipe.band_family`
is declarative for the same reason the LRSI recipes declare their timeframe: the
recipe id and the levels that produced a row can never drift apart.
`build_outcomes` takes `variant_bands_by_occurrence` and picks the map from the
recipe. **A variant recipe with no challenger bands gets NO bands** — it walks
`plain_no_target` and the row says so — rather than falling back to the
champion's levels, which would file the champion's answer under the challenger's
name.

**Its own `outcome_definition_id` is a fence.** The twin writes
`band_variant_v1`. Every existing reader filters on `house_default_v1`
(`queries.slice_readout`), so a challenger row cannot wander into an aggregate
computed over champion levels. Row identity is
`(occurrence_id, recipe_id, outcome_definition_id)` and the recipe id already
differs, so the second id buys ISOLATION rather than uniqueness — that is the
reason for it.

**Registered before any outcome.** `trial_ledger` gains
`swing_house_variant_v1_twin`, status `collecting`, with the trader's
authorization, the declared floors (≥ 20 forward sessions counted from the first
session carrying BOTH families) and three failure modes — the first of which is
the one that will actually happen: the two families are missing on different
populations, so an unpaired comparison would measure coverage and report it as
edge.

**Which is why `band-coverage --compare` pairs.** One table, per knowledge
bucket, both recipes as adjacent columns, over the occurrence ids that have a row
under BOTH. An occurrence missing under either is counted on a `not_paired` line
and is in NEITHER recipe's numbers. Win rate is over RESOLVED
(TARGETED + STOPPED — an OPEN row has not answered the question) with
`swing_headline`'s Wilson lower bound, the ONE Wilson for every trader-facing win
rate.

**Every average on that table names its own denominator** (reviewer advisory,
2026-09-05). Mean net R and the win rate do NOT share an n: a row carries a
`net_r` when its walk finished — TARGETED, STOPPED, EXPIRED, AMBIGUOUS_BAR — and
carries none when it did not (OPEN, TRUNCATED), while the win rate counts only
TARGETED + STOPPED. So `mR n` is a printed COLUMN rather than something a reader
infers, and the header says which rows each number is over. On the packet's own
fixture the twin's mean of −1.61R rests on ONE row of three, which is exactly the
case a single shared n would have misrepresented. For the same reason `--compare`
and `--recipe` are **mutually exclusive at argparse** rather than one silently
overriding the other: a run that accepted both would print a table the operator
did not ask for and give no sign of it.

**Still shadow.** Nothing here promotes anything. The bands are almost all
`reconstructed` (BD-99), which is research evidence and never promotion evidence,
and `docs/AVWAP_BAND_VARIANT_STUDY.md` T4's three criteria — not this report —
decide.

**Reopen if** the champion's σ is ever promoted or replaced (which decision 0008
and plan.md sec 5 forbid without a sec 7 decision), or if a THIRD band family
appears — at which point `band_family` should become a lookup of column prefixes
rather than a two-way branch.
