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

Status of the build: Phases 0-8 landed (code); the 20-session pilot is a live run that has not happened. Test baseline and
branch live in [`SOL_PROGRESS.md`](../SOL_PROGRESS.md). The 2026-08-04 review
(`docs/RESEARCH_WAREHOUSE_REVIEW_2026-08-04.md`) repaired the outcome engine
(BD-53..BD-57) plus four mechanical defects (Windows lock probe, protected
spool shedding, spool re-seal dedup, capture reconnect).

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

## BD-44 — OPEN: no detector adapter yet

**Not a decision — a declared gap.**

`record_occurrences` accepts a documented detection dict (symbol, canonical
setup id, side, timeframes, status, trigger, geometry, detector version).
Nothing yet reads the champion's tracker or scan output and produces those
dicts, so Phase 6 proves the identity rules, the recipes, and the outcome
arithmetic against constructed detections rather than live detector output.

Phase 2 already wraps the setup tracker into bronze, which is the intended
source. The adapter is deliberately not guessed at here: mapping tracker fields
onto detection fields is a decision about what the champion *means*, and
getting it wrong would silently mislabel every occurrence.

**Where.** `occurrences.py` (input contract in the module docstring and
`build_occurrence_row`).

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

**Not verified here.** The pin is declared; installing it on the Windows desk
(Python 3.14) has not happened. If that install fails, remove the pin and stay
on pyarrow — nothing else changes.

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

---

## Open items for Sol / Fable

Things this build deliberately left for a human decision or a live check.
Each is already stated in its own BD entry; this is the short list.

| # | Item | Where | What is needed |
|---|---|---|---|
| 1 | **Nothing calls the tee during a live session.** The service and tiles now exist; the remaining wiring is (a) the GUI handing BounceBot's `latest_bars` to `capture_m5_tee` each cycle and (b) the Health page rendering the six tiles. Capture otherwise runs only from a manual build job, so the pilot cannot start. | BD-20, BD-52 | Wire both, then start the 20-session pilot |
| 1b | **The 20-session pilot has not run** — it is a live-desk activity, not code. | BD-52 | Run it once capture is live; log the sec 5.6 measurements |
| 2 | **`ib_capture.build_ib_transport` is unverified** — the real ibapi client has no offline test and no broker-marked live run. | BD-25 | One live run on the desk before the pilot leans on it |
| 3 | **`exploration_cohort.txt` is empty** — the fixed 30 symbols define part of the research denominator, so no agent invented them. | BD-12 | Trader supplies the list (confirmation register item 5) |
| 4 | **Two favorite-zone definitions are the builder's** — `first_dev_touch_order` and `band1_rejection_strength`. | BD-32 | Confirm or amend; a change is a `feature_set_version` bump |
| 5 | **Production context columns are null until Phase 6** wires the bounce-ledger join. | BD-33 | Nothing now; noted so the gap is not mistaken for "no signal" |
| 6 | **DYNAMIC and EOD session VWAP are not yet captured** — only STANDARD. | BD-34 | Wrap the other two champion paths when their consumers need them |
| 7 | **Unscheduled exchange closures** cannot come from calendar rules; they appear as sessions with no bars. | BD-26 | Add a dated override list if one ever occurs |
| 8 | **No detector adapter yet** — `record_occurrences` takes a documented detection dict, but nothing reads the tracker/scan output and produces those dicts. Phase 6 proves the identity and outcome logic; the adapter is the remaining wiring. | BD-44 | Build the tracker→detection adapter (with Phase 2's bronze tracker wrap as the source) |
| 10 | **DuckDB pin is unverified on Windows/3.14** — the wheel exists, the install has not been run on the desk. | BD-45 | Install once on the desktop, or drop the pin and stay on pyarrow |
| 9 | **Bounce link is a time window** (symbol + session + ±60 min), not an explicit key. | BD-43 | Confirm the window, or add an occurrence link to the bounce ledger |

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
