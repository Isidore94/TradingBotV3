# Journal Reliability and UX Plan — Phase 0.5 R7

**Status: BUILT 2026-08-15; release-candidate fix pass 2026-08-16; live gates owed.**
Trader-directed packet (2026-08-15 desk request: "the journal misses trades, has
trades open — not acceptable; I need this for tax purposes too"). **Build
authorized immediately by a second trader redirect later on 2026-08-15**: branch
`phase05-r7-journal-reliability-ux` is cut from the tip of
`phase05-r2-focus-gating-strength-board` (the P0.7 merge has not happened; the
prior "after the merge" gate is superseded in writing, same pattern as R1/R2).
The desk keeps running the R2 branch via its scheduled task until Monday's
validation day passes — never switch the desk branch without disarming that task.
Merging R7 therefore brings the testing week, R1, R1.1, R2 and R7 together, and
R1/R2's owed live proofs are inherited unchanged. This packet builds **before**
R8 (`docs/WEEKEND_PREP_PLAN.md`) because R8's walk-away and tag-review steps read
the journal. The full planning conversation and diagnosis live in the session plan
of 2026-08-15; root causes below were verified against code at head of
`phase05-r2-focus-gating-strength-board`.

## 1. Purpose and product boundary

Tax-grade capture and analysis of every trade in the trader's IBKR and Questrade
accounts, plus a Journal tab that mimics the TradesViz/TraderSync basics: fast
edits and notes, excellent broker integration, good tagging, per-setup performance,
walk-away analysis, per-account P&L with tax-free/taxable separation, and full
commission/fee accounting.

Decision-support only; no order execution. The journal is capture/analysis-side —
nothing here feeds detectors, scores, gates, watchlists, or alerts. No fenced
detector/scoring/alert file is touched.

## 2. Invariants (locked)

- **I1 Completeness**: every broker execution appears exactly once in
  `raw_executions`, keyed by stable `execution_uid = BROKER:account:exec_id`.
- **I2 Coverage honesty**: a (broker, account, day) is COVERED only when a
  successful import actually spanned that day. Gaps are visible, never inferred.
  IBKR socket pulls never mark coverage (they only see the current TWS session).
- **I3 Append-only corrections**: raw broker rows are never destructively edited
  or deleted. Every human correction is an append-only `trade_adjustments` record
  applied deterministically at rebuild; undo = a superseding record. Precedent:
  `opportunity_events`.
- **I4 Annotation survival**: `rebuild_trades` may never orphan
  `trade_annotations` or reviews. The re-key pass plus `trade_aliases` are part of
  rebuild and are tested (zero-orphan is a permanent SQL test).
- **I5 Currency honesty**: FX rates are booked once per (date, currency) at import
  time from the Bank of Canada; never fetched at render. A missing rate renders as
  "unconverted", never as 0 and never silently native.
- **I6 Tax separation**: taxable and tax-free accounts are never silently blended;
  any blended view carries an explicit badge.
- **I7 Trader-owned fields** (tags, notes, reviews, planned stop/risk, tax-status
  overrides) are never written by any import/rebuild path.
- **I8 Nightly job**: runs only via the `journal_import` runner slot inside the
  off-hours window (`scripts/ai_jobs`); no new timer, no new thread owner, no new
  ntfy sender. Zero-execution night = `ok`.

## 3. Root-cause register (diagnosis 2026-08-15, file:line verified)

Missed trades:
- **A1** Qt import service hard-codes `include_ibkr=False`
  (`scripts/ui/services/journal_import_service.py:26-33`); the Qt panel exposes no
  IBKR/Flex/backfill control anywhere.
- **A2** IBKR socket path: `ExecutionFilter` with no time filter
  (`scripts/journal_importers.py:417-429`) sees only the current TWS session;
  `target_date` is ignored (`scripts/journal_runner.py:111-115`). Any day the
  import doesn't run before the TWS nightly reset is permanently lost to this path.
- **A3** the EOD slot treats `failed` as final and never retries
  (`scripts/master_avwap_mini_pc.py:109, 673-682`) — permanent one-day holes.
- **A4** socket timeout/partial returns silently as OK (`journal_importers.py:422-429`).
- **A5** one bad account/chunk discards the whole pull — executions accumulate in a
  local list, store write happens after return (`journal_importers.py:345-375`,
  `journal_runner.py:68-70`).
- **A6** `import_runs` records no date coverage (`scripts/journal_store.py:176-184`)
  — gaps are structurally undetectable.
- **A7** `QuestradeImporter.get_activities` is written but never called
  (`journal_importers.py:286-295`); the Flex parser reads only Trade/TradeConfirm
  (`:549`) — option expiries/assignments/transfers are never captured.
- **A8** env-var Questrade token: the rotated single-use refresh token is discarded
  (`journal_importers.py:254-255`) — auth breaks after the first refresh.
- **A9** the real backfill (`journal_runner.py:168-251`, incl. IBKR Flex at
  `:208-229`) is CLI-only; UI pulls cap at 31 days.
- **A10** `_is_ibkr_client_id_conflict` operator-precedence bug: any error text
  containing "326" is misread as a client-id conflict (`journal_importers.py:65-67`).

Stuck-open trades:
- **B1** pure execution netting; CLOSED only at qty≈0 (`journal_store.py:438-493,
  704`); `CLOSED_PARTIAL` is never produced; no reconciliation against broker
  positions exists anywhere.
- **B2** a missing opening fill fabricates a phantom inverse trade
  (`journal_store.py:463-468, 624`).
- **B3** unstable group key: Questrade `security_type` falls back to
  `listingExchange` (`journal_importers.py:305`); socket vs Flex vocabularies and
  option symbols differ; MANUAL fills are keyed `broker="MANUAL"` and never attach
  (`journal_importers.py:637-638`).
- **B4** `execution_uid` embeds the timestamp (`journal_importers.py:323/447/562`)
  — the same fill from socket and Flex dedupes as two fills and doubles the
  position.
- **B5** unparseable timestamps silently become `now()`
  (`journal_importers.py:123-140`); ibapi 10.x tz-suffixed format unhandled.
- **B6** `trade_id` is content-derived from the first execution + sequence
  (`journal_store.py:625`); `rebuild_trades` deletes trades but not
  `trade_annotations` (no FK; `:496-498, 251-256`) — a backfill re-keys trades and
  orphans every tag/note/review.
- **B7** no correction API exists at any layer.
- **B8** multi-currency P&L summed unconverted (`journal_store.py:733`,
  `journal_analytics.py:356-377`).

## 4. Data model (schema v2 → v3; one migration; machinery is net-new)

`initialize_schema` reads `meta.schema_version`; if `< 3`, take a timestamped file
copy `trade_journal.sqlite3.bak-v2-<ts>` beside the DB, then run `migrate_to_v3`.
New `scripts/journal_migrate.py` CLI supports `--dry-run` against a copy, printing
a full report (duplicate collapses, re-key mapping, orphan check) without touching
the live file. DB: `<shared_home>/data/runtime/trade_journal.sqlite3`.

New tables:
- `import_coverage(broker, account_number, day, status
  COVERED|FAILED|PENDING|NO_SESSION, source, import_run_id, attempts, message,
  updated_at, PK(broker, account_number, day))` — the coverage ledger.
- `fx_rates(rate_date, currency, rate_to_cad, source='BOC_VALET', effective_date,
  fetched_at, PK(rate_date, currency))` — `effective_date` records the actual BoC
  observation date when a weekend/holiday rate is carried back (CRA-acceptable).
- `trade_adjustments(adjustment_id uuid PK, target_kind EXECUTION|TRADE_GROUP,
  target_uid, action VOID_EXECUTION|EDIT_EXECUTION|ADD_EXECUTION|FORCE_CLOSE|
  REASSIGN_GROUP, payload_json, reason NOT NULL, source, superseded_by,
  created_at)`.
- `trade_aliases(old_trade_id, new_trade_id, reason, created_at,
  PK(old,new))` — identity history; `opportunity_events` stay immutable and
  resolve through aliases transitively.
- `cash_transactions(txn_uid PK, broker, account_number, txn_date, activity_type
  FEE|INTEREST|DIVIDEND|FX|OTHER, description, symbol, amount, currency, raw_json,
  imported_at)` — Questrade activities + Flex CashTransactions.

Column additions (all additive `ALTER TABLE`):
- `import_runs` += `coverage_start`, `coverage_end`, `account_number`, `trigger`.
- `accounts` += `tax_status` (`TAXABLE|TAX_FREE|TAX_DEFERRED`, seeded from
  Questrade account_type: TFSA→TAX_FREE, RRSP/RESP/LIRA→TAX_DEFERRED,
  Margin/Cash→TAXABLE, else blank) and `tax_status_source` (`auto|trader`);
  the import upsert never clobbers a `trader`-sourced value (I7).
- `raw_executions` += `source` (`QT_API|IBKR_SOCKET|IBKR_FLEX|MANUAL|CSV`),
  `multiplier`.
- `trades` += `net_pnl_cad`, `fx_rate`, `fx_rate_date`, `reconcile_status`
  (`''|NEEDS_REVIEW|FORCED_CLOSED`), `anchor_execution_uid`; status set gains
  `CLOSED_PARTIAL`.
- `trade_annotations` += `planned_stop`, `planned_risk`, `planned_entry`,
  `risk_source` (`manual|alert_prefill`) — R fields live here because this table
  survives rebuilds and is trader-owned.

Identity fixes:
- **execution_uid → `BROKER:account:exec_id`** (drop symbol+timestamp at
  `journal_importers.py:323/447/562`; the manual path keeps its uuid). Migration
  groups existing rows by new uid; on collapse keeps the richest row
  (IBKR_FLEX > IBKR_SOCKET, detected from raw_json shape; Questrade keeps newest
  `imported_at`), reports every kept/dropped uid. Doubled socket+Flex positions
  collapse here.

  Three choices this left open, **decided at build time (§9 step 2, 2026-08-15)**
  and recorded here because each is load-bearing:

  1. **`BROKER` in a uid is the token the importers already emit** — `QT`,
     `IBKR`, or the manual broker string — not the long `QUESTRADE` spelling of
     the `broker` column. Rewriting `QT` to `QUESTRADE` would churn every
     Questrade uid for no gain and invalidate every uid a human has already read
     in a report.
  2. **The same source-precedence rule runs at import time, not only in the
     migration.** Once the uid stops embedding the timestamp, a desk-hours socket
     import and that night's Flex pull land on the *same row*, so "which wins" is
     a live question and not only a one-time one. `upsert_executions` refuses to
     let a poorer source overwrite a richer one, so a socket import can never
     erase the commissions, fees and netCash a Flex row already carries.
  3. **A broker row with no execution id gets a deterministic surrogate** —
     `PREFIX:account:auto-<sha256 of order_id|symbol|timestamp|side|qty|price>` —
     not a random uuid. These are the stable fill discriminators; commission,
     fees, auxiliary activity ids and the full raw payload are deliberately not
     identity because a broker correction or payload-schema addition must update
     the same fill rather than mint a duplicate.
  4. **Legacy Questrade order-id uids are re-keyed before collapse.** V2 used
     `orderId` when a Questrade execution had no id, but one order may contain
     several partial fills. The v2→v3 migration recognizes that provable shape
     (Questrade row, no execution id in the raw payload, uid id equal to
     `order_id`) and assigns each row the discriminator hash above before
     grouping. The report names how many legacy order-id groups and rows it
     found. Later pulls therefore update those survivors under the same hash;
     they cannot reinsert beside one order-id-keyed row and double-count it.

- **trade_id anchored to the opening execution uid**, plus a re-key pass inside
  `rebuild_trades`: snapshot annotated trade_ids + their leg uid sets before the
  DELETE; after assembly, map old→new by largest execution-uid overlap; UPDATE
  `trade_annotations`, append `trade_aliases`. Ambiguous mappings are listed in
  the migration report for trader review, never guessed silently.

## 5. Import pipeline

Source-of-truth policy:
- **IBKR: Flex primary.** Nightly, backfill, and coverage all use the Flex web
  service (already implemented: `journal_importers.py:532-632`, settings
  `journal_ibkr_flex_token` / `journal_ibkr_flex_query_id`). Coverage is marked
  from the FlexStatement's own `fromDate`/`toDate` (parser gains statement
  metadata). The socket importer remains a desk-hours convenience with the same
  stable uid (dedupes cleanly) and never marks coverage.
- **Questrade: executions API primary for trades** (per-day/range, 31-day chunks —
  already built); **activities API additive** for fees/dividends/interest/expiries
  and as a completeness cross-check (wire the existing `get_activities`).

Fixes in dependency order (each a commit; see §9):
1. Hygiene: A10 precedence fix; B5 strict timestamp parsing (ibapi 10.x tz suffix
   via zoneinfo; importer-strict mode quarantines the row with raw payload instead
   of stamping `now()`); A4 raise on timeout-without-`execDetailsEnd`.
2. B4 uid migration (§4).
3. B3 group-key normalization: deterministic `security_type` classification (stop
   the `listingExchange` fallback), one vocabulary across socket/Flex, options
   grouped on OCC-style symbol in both paths; the manual-execution dialog gains
   real broker/account pickers; `REASSIGN_GROUP` adjustments cover historical
   MANUAL rows.
4. Assembly (characterization fixture FIRST — §9 step 0): `CLOSED_PARTIAL`;
   missing-opening-fill produces a `SYNTHETIC_OPEN`-marked leg +
   `reconcile_status='NEEDS_REVIEW'` instead of a phantom inverse trade;

   **Narrowed at build time (§9 step 4, 2026-08-15).** Only the *unambiguous*
   missing-opening-fill is flagged: a fill that closes more than the journal
   knows is open, whose leftover is itself the proof that an opening fill is
   missing. A plain sell with no position open is genuinely ambiguous — a real
   short entry and a sale of shares bought before the import window are
   indistinguishable from the execution alone — so it assembles as an ordinary
   short and is left to fix 9's reconciliation, where the broker reporting flat
   against a journal that says short is the evidence assembly cannot have.
   Flagging every short would fill the review queue with correct trades.
   **Trader-approved as built, 2026-08-15** — a decided narrowing, not an open
   item.


   adjustments applied deterministically (VOID skips, EDIT overlays, ADD injects,
   FORCE_CLOSE injects a synthetic closing fill and stamps `FORCED_CLOSED`);
   anchor-based trade_id + re-key pass.
5. `trade_adjustments` store API (`record_adjustment`, `list_adjustments`,
   supersede/undo).
6. Partial persistence: per-account/per-chunk try/except, `upsert_executions`
   after each chunk, per-(source, account) `import_runs` rows with coverage span;
   a failed chunk marks only its own days FAILED. New `scripts/journal_coverage.py`:
   `mark_coverage`, `find_gaps` (skips NO_SESSION via `market_calendar.is_session`),
   `self_heal(max_days_per_night=62, max_attempts_per_day=5)` — retries FAILED,
   backfills gaps, bounded per night. `journal_inception_date` local setting per
   broker seeds the ledger horizon.
7. A7: wire activities → `cash_transactions` (Trades-type activities are
   cross-check only — executions API stays authoritative, no double-count); extend
   `parse_ibkr_flex_statement` for `OptionEAE` (synthetic executions so option
   positions actually close), `OpenPositions` (reconciliation input, not stored as
   executions), `CashTransactions`.
8. FX: new `scripts/journal_fx.py` — BoC Valet
   (`/valet/observations/FX{CUR}CAD/json`, free, no key); `ensure_rates` fetches
   only missing (date, currency) rows, carries the prior business-day observation
   onto weekends/holidays (storing `effective_date`), never blocks the import
   path. `rebuild_trades` finalize books `net_pnl_cad` from the stored table only
   (I5); CAD books 1.0. Analytics gain an explicit `pnl_key` and stop
   cross-currency summing (B8).
9. Reconciliation: new `scripts/journal_reconcile.py` — Questrade
   `GET v1/accounts/{id}/positions` (new `get_positions`) + Flex OpenPositions vs
   the journal's net-open per (broker, account, instrument); mismatch →
   `NEEDS_REVIEW` + an append-only summary run row; journal-open-but-broker-flat
   also produces a *suggested* FORCE_CLOSE adjustment the trader confirms in the
   UI — never auto-applied. Runs at the end of the nightly slot and on demand.
10. A8: token precedence inverted — local settings win; env var is a first-boot
    seed only; status line warns if env vars remain set.

## 6. Nightly job (promotion of the queued P3.3 slice — trader-approved 2026-08-15)

`run_nightly_journal_import()` in `journal_runner.py`: Questrade last-7-days
ranged pull + activities → IBKR Flex pull (when configured) → `self_heal` →
`ensure_rates` → `rebuild_trades` → `reconcile`. Registered as
`JobSlot("journal_import", reserve_minutes=5, max_attempts=3)` inserted FIRST in
`ai_jobs.runner.default_slots()` — the slate-order exception sanctioned by
`docs/LOCAL_AI_AUTOMATION_PLAN.md` §6.4c, whose design decisions (slot-first,
Flex-not-socket, token race stated-not-solved, one writer, zero-execution night =
`ok`) are honored verbatim. The §6.4b ticker-briefs live proof is about the
inference layer, not this seconds-scale slot; the trader's 2026-08-15 promotion
supersedes the "after 6.4b proof" ordering — recorded here honestly rather than
silently. `ai_jobs` stays out of the frozen bundle; `run_nightly_journal_import`
lives in `journal_runner` so the desk Health tab can also invoke it manually.

The retired `master_avwap_mini_pc.py` EOD slot is untouched; ownership transfers
to this slot. Open operational item: audit Task Scheduler for a stale entry that
could still fire the old EOD import (harmless double-write, but confusing).

**Schema-preparation decision (release-candidate fix pass, 2026-08-16):** the
Journal GUI gates on the persisted `meta.schema_version`, not whether this Python
process has already constructed a store. An already-v3 database opens normally
after every fresh launch. A non-empty pre-v3 database must be prepared by the
trader in the GUI so the dry-run can be reviewed and the backup/migration remains
trader-present. The nightly slot refuses that database and records FAILED; it does
not auto-migrate it. A brand-new absent database may still be initialized by the
nightly slot because there is no existing data to back up or review.

## 7. UI (Qt only; the Tk tab stays legacy and untouched)

`JournalPanel` becomes a shell over a `QTabWidget`; sub-tabs extracted to a new
`scripts/ui/panels/journal/` package (`trades_tab.py`, `calendar_tab.py`,
`analytics_tab.py`, `health_tab.py`, `fees_tab.py`). All data access stays behind
the extended `ui.services.journal_feed` facade.

Shared header above the tabs: checkable **account tree** grouped
Taxable/Tax-free/Tax-deferred/Unlabeled from `accounts.tax_status`, with a blended
warning badge when the selection spans tax groups (I6); **currency toggle**
CAD/USD/Native; **date-range filter** (7d/30d/QTD/YTD/All + custom;
`list_trades` gains `date_from`/`date_to`).

> **DEFERRED — release-candidate reconciliation, 2026-08-15:** USD display is
> currently exact only for USD-native trades. A selection containing CAD or
> another currency refuses the USD total instead of relabeling native money;
> true booked USD conversion remains future work. The Calendar ships the month
> grid but not the promised pyqtgraph year heatmap. Analytics ships its equity
> curve and non-exclusive grouping table, but the per-setup/account bar charts,
> day-of-week/time-of-day charts, and R-distribution/expectancy charts (plus
> chart-specific CSVs) remain deferred. These are visible product gaps, not
> release-candidate behavior.

- **Trades**: existing table + KPI tiles (computed over the filtered/converted
  selection). Detail pane adds: R-fields group (`planned_entry/stop/risk`
  editable, live R readout = `net_pnl_cad / planned_risk`, "Prefill from alert" —
  `journal_feed.suggest_planned_risk` joins symbol+direction+opened_at±1 session
  against armed-alert review events carrying entry/stop; unique match only; never
  overwrites trader values); legs view with source and SYNTHETIC/adjustment
  markers; auto-tag review pane (`list_auto_tag_candidates` → accept appends tags
  + `record_tag_corrections`); corrections launcher (void/edit/add/force-close →
  `trade_adjustments` with mandatory reason → rebuild; per-trade audit list);
  NEEDS_REVIEW banner with the reconciliation delta.
- **Calendar**: month grid from `calendar_pnl_by_day`; day click filters the
  Trades tab. The year heatmap is deferred above.
- **Analytics**: pyqtgraph equity curve (cumulative `net_pnl_cad`) plus the
  grouping table described below. The additional charts and their exports are
  deferred above. `journal_analytics`
  fixes: per-tag expansion (replace `_first_setup_tag`; multi-tag trades count in
  every tag bucket, noted as non-exclusive), and the setup group split into
  **"my setups"** (`setup_tags`) vs **"auto tags"** (`auto_tag_summary`) instead
  of the silent fallback. Walk-away section runs `journal_walkaway` in a worker
  for the selected range and renders its outputs. **R7 adds the additive
  `since`/`until` kwargs to `run_walkaway_analysis` and its loaders** (defaults
  preserve current behavior; characterization test); R8 consumes them.
- **Health**: import-run history; coverage grid (accounts × recent days, colored
  by status) with "Backfill gaps" / "Retry failed" buttons (worker-run
  `self_heal`); reconciliation flags + confirm-force-close flow; nightly-slot
  status read-only from the job ledger; Broker Sync drawer extended with IBKR
  socket settings, **Flex token/query-id fields**, a full-backfill button, and
  the Questrade env-var warning. The import service gains
  `pull(days, include_ibkr, include_flex)` + `backfill()` (closes A1/A9).
- **Fees**: per-account × per-currency totals of trade commissions+fees plus
  `cash_transactions` by activity type, date-ranged, `export_fees_csv`.

Models: `JournalTrade` += `net_pnl_cad`, `currency`, `reconcile_status`,
`r_multiple`, `tax_status`; the table model gains currency-aware columns and a
NEEDS_REVIEW row tint.

## 8. Trader one-time setup (Flex runbook) — DONE 2026-08-15

**Completed and live-verified by the trader.** The token and query id are in
machine-local settings; a read-only verification returned **372 trades** over
the 365-day window across **both** IBKR accounts, with all four sections
present. Questrade's rotating-token chain is stored and authenticating, serving
accounts TFSA 51830546 and Margin 29347316. The runbook below is kept for the
next time a query has to be rebuilt.

Two constraints remain while the live migration is still owed: **read-only
against the brokers, and no writes to the live journal database**; and **no
needless Questrade token refreshes** — the chain is single-use rotating and
anchored on this desk.


IBKR Account Management → Performance & Reports → Flex Queries → new **Activity
Flex Query** with sections: **Trades** (execution level; include at minimum
accountId, symbol, assetCategory, currency, buySell, quantity, tradePrice,
ibCommission, other commissions/fees, netCash, ibExecID, ibOrderID, dateTime,
multiplier, underlyingSymbol, expiry, strike, putCall), **Option Exercises,
Assignments and Expirations**, **Open Positions**, **Cash Transactions**. Period:
Last 365 Calendar Days. Format XML, date format `yyyyMMdd;HHmmss`. Then Settings →
FlexWeb Service → enable → generate token. Paste token + query id into
Journal ▸ Health ▸ Broker Sync. History older than 365 days: run one custom-period
Flex report from Account Management and use the "Import Flex file" button.

## 9. Build order (commit-sized; tests per step)

0. Characterization fixture: script-generated fixture DB (scale-ins, partials,
   shorts, options with multiplier, a socket+Flex duplicate pair, a
   missing-opening-fill case, a MANUAL fill, multi-currency) + golden JSON of
   current `rebuild_trades` output, asserted bit-for-bit BEFORE any assembly
   change. Later intended diffs update the golden with an explicit note.
1. Hygiene (A10, B5, A4). 2. Migration machinery + v3 DDL + uid migration +
   `journal_migrate.py`. 3. Group-key normalization. 4. Assembly changes + the
   never-tested cases (missing middle execution, stuck-open, socket-vs-Flex
   dedupe, group splitting, annotation survival across a backfill that inserts an
   earlier opening fill). 5. Adjustments API. 6. Partial persistence + coverage
   ledger + self-heal. 7. Activities + Flex OptionEAE/OpenPositions/
   CashTransactions. 8. FX + analytics fixes + walkaway `since`/`until`.
   9. Reconciliation. 10. Nightly slot. 11. UI Trades tab + shared header.
   12. UI Calendar + Analytics. 13. UI Health + Fees. 14. Governance close-out
   (CHANGELOG, `docs/README.md`, `WISHLIST.md`, `plan.md` narrowing,
   `CURRENT_CHECKPOINT.md`).

New top-level modules (`journal_migrate`, `journal_coverage`, `journal_fx`,
`journal_reconcile`) are statically imported from `journal_runner` /
`journal_feed` so the frozen bundle collects them; frozen rebuild + selftest
before merge.

**Trader-present steps**: Flex setup (§8) before step 7 goes live; account
tax-status labeling after step 11; first live migration + full backfill (auto
backup + dry-run report reviewed; spot-audit ≥10 trades against statements);
reconciliation-week sign-off; Questrade env-var cleanup if set.

## 10. Exit gates

Deterministic: full suite green from the current baseline (see
`CURRENT_CHECKPOINT.md`), smoke 7/7, source selftest green, migration dry-run
report reviewed with zero orphaned annotations.

Live:
1. After full backfill, `import_coverage` shows COVERED or NO_SESSION for every
   session day since inception for every account.
2. Trade counts and commission totals reconcile to one monthly statement per
   broker to the cent.
3. One full week of nightly reconciliation with zero unexplained position
   mismatches on both brokers (every mismatch fixed upstream or explained by an
   adjustment record).
4. Zero rows in `trade_annotations` without a matching trade (permanent SQL
   test).
5. CAD totals spot-checked against published BoC rates for 3 dates.
6. ≥5 consecutive nightly `journal_import` ledger entries with coverage advancing
   and at least one observed self-heal of a failed/gap day.

## 11. Risks and open items

- Flex polling (12×5 s) may be short for a 365-day statement — make
  attempts/interval settings. Flex web service caps at 365 days; older history via
  the one-time file import (§8).
- Questrade activities schema varies by account era — classifier defaults to
  OTHER and keeps raw_json; cross-check-only role bounds the blast radius.
- BoC same-day rate unavailable until ~16:30 ET; non-USD/CAD pairs may lack daily
  observations — fall back to unconverted + warning (I5).
- Ambiguous annotation re-keys are listed for trader review, never guessed.
- Stale Task Scheduler EOD entry audit (§6).
- R-prefill joins alert events (symbol+time) to trades (account+symbol) —
  same-symbol re-entries in one day are ambiguous; prefill only on unique match.
- Desk Modern Standby can shorten the nightly window; self-heal on the next
  firing is the mitigation. GUI-vs-nightly SQLite write collision: accepted per
  §6.4c decision 4 — surfaces as a FAILED run and self-heals; no new locking.

## Deferred visuals — BUILT 2026-08-18

R7 deferred three things at build time: true non-USD conversion, the Calendar
year heatmap, and additional Analytics charts. The last two landed under the
trader's 2026-08-18 integration redirect. **True USD conversion stays deferred**
— the FX table books CAD only, and inventing a rate is exactly the dishonesty
the currency refusal was built to prevent.

**Analytics per-group charts, with honest n and a CSV underneath.** A group
picker over the breakdowns the tab already computed (my setups, auto tags,
account, broker, symbol, direction, the three regimes), a bar chart of net by
bucket, and an export of exactly what is charted. Three rules make it safe to
read:

1. every bar carries **n as closed trades**, and a bucket under the thin-sample
   line says "thin" on its own label — a two-trade setup must not look like a
   finding;
2. a bucket whose total is `None` is **excluded, not zeroed** — `None` means
   "mixed currencies with unconverted rows", and a zero bar would claim the
   setup broke even;
3. what the bar cap and the exclusions dropped is **counted and printed**, and
   the overlapping groups (my setups, auto tags) say that they overlap and do
   not sum to the headline.

**The Calendar year heatmap.** A pyqtgraph image of the year, diverging
red→white→green, **centred on zero and scaled to the largest absolute day**.
Scaling to the raw min/max would make a good year look mediocre purely from
where the extremes fall. A day with no trading is **blank, not a break-even
colour** — the matrix carries `None` for it and the image carries NaN — because
a flat day and a day the trader did not trade are different facts, and a heatmap
that paints them alike invents a hundred break-even sessions a year. The numeric
grid stays underneath, and both surfaces still filter the Trades tab on a click.

The six live gates in this spec are unchanged and still owed; none of the above
touches the store, the migration or the reconciliation path.

## Nightly-slot honesty and the Questrade chain surface — BUILT 2026-08-24

Two packets from `docs/analysis/AI_LAYER_REVIEW_2026-08-24.md` §5 landed on
`testing-week-2026-08-24` (`5350361`, `40d7d3a`); neither touches the store,
the migration, or reconciliation itself.

**AI-P3 — the nightly slot has something to say.** The review's stated defect
was REFUTED by reproduction: reconcile mismatches never marked a run FAILED and
never burned attempts — only a reconcile exception sets `had_errors`
(`test_reconcile_mismatches_do_not_make_a_successful_import_a_failure` is the
kept regression). The real defect: the runner records `outcome["reason"]` and
this job returned only `messages`, so every `journal_import` ledger row ever
written was mute — which is how the dead Questrade chain sat undiagnosed for
five nights. Now `run_journal_backfill` returns `failures` beside `status`
(every `had_errors` site names itself), and `_nightly_reason` builds the ledger
line from what the night measured — imported/rebuilt/self-heal/reconcile
counts, first three failures named, the dropped count printed
(`NIGHTLY_REASON_FAILURE_LIMIT = 3`). A value not measured is absent, never
zero: a night whose reconciliation did not run says "reconcile skipped", not
"0 mismatches". §6's own precedent governs ("zero-execution night = ok").
Advances gates 3 and 6, which remain owed as live proofs.

**AI-P4 — a dead Questrade chain is visible on the desk.**
`scripts/journal_health.py` reads `journal_questrade_expires_at` and the
coverage ledger's most recent oauth-pattern failure (by attempt time, not
coverage day — a year-spanning backfill marks every session day FAILED from
one break, so the coverage day would misdate the outage) and renders
UNHEALTHY/stale/not-configured states in the Journal Health tab and System
Health (`operations_audit`), with the repair step in words: paste a fresh
refresh token into Journal ▸ Health ▸ "Questrade refresh token". Absent
settings read "not configured", an unreadable database reads unknown — never
healthy-by-default. Naive/aware stamps are normalized by attaching the aware
side's zone to the naive side, never by stripping. Advances gate 1 (the full
Questrade backfill becomes reachable and its regressions visible); the gate
itself still owes the trader's portal action and the covered-since-inception
proof.
