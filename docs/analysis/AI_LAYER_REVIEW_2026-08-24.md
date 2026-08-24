# AI-layer review — 2026-08-24

**Read-only analysis; hand-committed and frozen.** Written on branch
`testing-week-2026-08-24` (tree at `97b6ae7`, identical code to
`phase05-integration-blitz`). No file under `scripts/` or `tests/` was modified
by this work. Every §1 number from the tasking brief was re-measured at the
source before use; differences are flagged inline as findings. Statistics
claims in this document are labelled `discovery` or `confirmation` per plan.md
Phase 0.7 ground rule 10.

---

## 1. Verdict

The AI layer's four-slot slate is healthy as machinery and lopsided in value:
the one deterministic grading slot (`veto_cohort_grading`, seconds per night, no
model) produced the only decision-relevant finding the layer has ever emitted,
while the two model slots (`ai_summary`, `ticker_briefs` — together all of the
window time and all of the model calls) have no measured reader and therefore
remain UNKNOWN, and the `journal_import` slot does real, read work but has never
once recorded `ok`. The binding constraint is **input poverty on the
trader-decision substrate**, proven twice over: the Questrade side of the
journal is entirely dark (0 of 142 attempted days covered, a dead single-use
refresh chain), and the annotation/tag-confirmation stream the enrichment
phases were designed to read holds 1 confirmed annotation and 1 correction
against 220 waiting candidates. It is not window contention (72 min used of an
8-hour window), not model misallocation (the medium tier is the correct derived
model; the large tier is properly retired), and not the absent digest ledger
(deliberately unbuilt, and its only authorized reader does not exist yet).

---

## 2. Per-slot ledger (Q1)

**Value test used.** A slot is **PROVEN** when reproduction shows (i) it wrote
an artifact that made a previously unanswerable question answerable, (ii) a
named reader actually consumed it, and (iii) its cost fit its reserve.
**REFUTED** when reproduction shows the artifact is unread-by-construction,
unreadable, or wrong. **UNKNOWN** when readership or content quality is
unmeasured. Defense: this is R10 ground rule 3's reproduction discipline applied
to value instead of defects, and it mirrors the layer's own §7.1 rationale for
the deterministic slot ("computable-but-unanswered" is the failure mode the
slate exists to close). A clean run log does not satisfy (i)–(ii).

All provenance below: ledger = `\\MINI-PC\Trading Bot Data\ai_store\logs\ai_job_ledger.jsonl`
(427,597 bytes, 72 rows, parsed with a `Counter` over `(job, status)`); journal =
`sqlite3.connect('file:C:/TradingBotData/data/runtime/trade_journal.sqlite3?mode=ro', uri=True)`;
cohort = `C:\TradingBotData\data\runtime\veto_cohort_performance.csv`; task =
`schtasks /query /tn "TradingBotV3 AI Jobs" /v /fo LIST`; per-firing logs =
`%LOCALAPPDATA%\TradingBotV3\logs\ai_jobs-YYYYMMDD.log`.

| # | Slot | Model? | Cost (measured) | Lifetime ledger record | Output / reader | Classification |
|---|---|---|---|---|---|---|
| 1 | `journal_import` | No — broker/network pull | 5-min reserve; 3 attempts/session, all 3 burned nightly | **0 ok / 9 failed / 3 skipped** (all 9 failures blank-reason — rows predate R10.0's `_failure_reason` floor, `scripts/ai_jobs/runner.py:350-384`) | `trade_journal.sqlite3` — read by the Journal page, Analytics, R8 walk-away + auto-tag pane | **Split verdict.** The import *machinery* is PROVEN: IBKR coverage 505 COVERED / 2 FAILED (`import_coverage` query), 187 trades / 534 executions exist and are read daily. The *slot as a health record* is REFUTED: it has never returned `ok` because three `had_errors` paths (transient Flex, Questrade `/activities` 400, 19 reconcile mismatches) mark a run FAILED *after* a successful import (R10.0 finding, CHANGELOG L617), and the Questrade half is dead (§3). |
| 2 | `ai_summary` | Yes — medium `gemma3:12b-tbv3ctx` | 20-min reserve | 14 ok / 7 degraded_no_narrative / 4 skipped, **plus 1 `correction` and 1 `manual_test` row the brief omitted** | Morning summary file | **UNKNOWN.** No evidence anywhere — ledger, docs, checkpoint — that the trader reads the output. 7 of 21 model-attempted runs degraded to no narrative. |
| 3 | `ticker_briefs` | Yes — medium tier, per symbol | **72.4 min** (4,342.1 s, started 2026-08-21T22:03:29-07:00), **65 model calls**, ledger reason "229 of 229 ticker(s) resolved… (65 model call(s), 0 reused, 0 failed)"; store `briefs/` 3,689 files / 20.55 MB | 9 ok / 11 failed / 9 skipped, **plus 1 degraded_no_narrative the brief omitted**. Finding: the 11 failures carry `session_date 2026-08-07` but `started_at` 2026-08-09T22:37 → 2026-08-10T03:30 — "all on 2026-08-07" in the brief is the session stamp, not the run date; all were "local provider returned invalid summary JSON" (the pre-`-tbv3ctx` 2,048-token context defect, since fixed) | Morning briefs file | **UNKNOWN, at the highest cost.** TB-2/TB-5 already cut 229 symbols to 65 calls, so the rationing works — but TB-5 measured pre-fix content at 96.2% roster noise and **post-fix content quality has never been measured**, and no reader is evidenced. Effectively the entire model bill of the layer. |
| 4 | `veto_cohort_grading` | **No** (test asserts the provider is never reached) | Seconds (file mtimes 23:15:51 → 23:15:54 on 2026-08-21); 5-min reserve | 2 ok / 1 skipped — exactly as briefed | `veto_cohort_outcomes.csv` + `veto_cohort_performance.csv` — read by the trader (this review), by the opt-in `trader_judgement` scope, and (once built) by R8's mirror-cohort pane | **PROVEN.** "Are my vetoes any good?" was computable-but-unanswered from the day `update_veto_cohort_outcomes` shipped until this slot became its first caller (LOCAL_AI_AUTOMATION_PLAN §7.1). The graded table in §3 below is the only decision-relevant finding this layer has produced. |

**Weekend ledger silence — REFUTED as a defect.** The brief asked whether two
nights of no ledger rows was a silent early exit. Reproduced: the scheduled
task fired every 30 min all weekend (per-firing logs `ai_jobs-20260822.log`,
`-20260823.log`, each ending "AI jobs complete (exit 0)"; task LastRunTime
2026-08-24 06:00:02, result 0). `session_date_for()` resolves weekend firings
to Friday 2026-08-21 (`runner.py:85-104`); all four slots were already terminal
for that session; the one-time `no_session` skip rows were written at
2026-08-21T23:31 PT and `_already_recorded_no_session` (`runner.py:120-132,
198-201`) deliberately downgrades every later firing to a debug log — the
documented anti-spam ("~27 rows a night", `runner.py:191-193`). Correct by
design. Corollary: **an empty ledger weekend is not a monitoring signal** and
should not be watched as one.

**Other re-measurement deltas from the brief** (all minor): `PRAGMA
user_version` is 0 — schema v3 lives in the `meta` table
(`schema_version='3'`, migrated 2026-08-23T21:45:24), so "schema is v3" holds
but not where a pragma check would look. The cohort CSV also carries three
small unbriefed cohorts (`compressed` n=3, `v1_incoming_trendline` n=5,
`v1_overhead_horizontal` n=2). `local_settings.json`'s
`journal_trader_tax_statuses` names a fourth account (IBKR U4867396) that has
no `accounts` row — flagged in §8 for the trader.

---

## 3. Binding constraint (Q2)

**PROVEN: input poverty on the trader-decision substrate.** Two concrete holes,
both reproduced:

1. **The Questrade half of the journal is dark.** `import_coverage`: QUESTRADE
   0 COVERED / 142 FAILED (plus 6 NO_SESSION). Dominant failure ×56:
   `500 Server Error … login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token=va8O…`.
   The brief's reading is confirmed and sharpened: the code *does* persist each
   rotated token (`journal_importers.py:401-402` saves the next refresh token
   immediately after every refresh), so this is not the A8
   discard-the-rotation defect — the stored chain itself is dead (the failing
   URL carries an older token than the one now stored, i.e. the chain forked or
   expired; `journal_questrade_expires_at` = 2026-08-22T00:30:04, expired).
   Recovery is a **trader portal action**: generate a new refresh token and
   paste it into the existing Journal ▸ Health field (§5, AI-P4). Two of the
   trader's four accounts, including a TFSA, contribute zero rows to every
   analytic the AI layer could read; the 7 consecutive nightly MISMATCH runs
   (last: "29 position(s), 19 mismatch(es), 19 trade(s) flagged", `import_runs`
   run 122) are at least partly this hole reflecting back.

2. **The annotation/confirmation stream is starved.** `trade_annotations` = 1
   row, `tag_corrections` = 1 row, `auto_tag_candidates` = 220 rows (direct
   counts). Phase 3 journal enrichment, the `trader_judgement` scope, and any
   future synthesis were all designed to read trader-confirmed structure; at 1+1
   confirmed rows there is nothing to read. The capture rail's veto stream is
   the exception — dense enough (78 graded picks at the 1-session horizon) that
   the deterministic grader produced signal — which is the existence proof that
   the constraint is input, not machinery: **where input was dense, the
   cheapest slot found the only finding.**

**The finding itself (labelled `discovery`, emphatically not `confirmation`):**
at the single matured 1-session horizon, SHORT vetoes graded superbly (n=35,
8.6% would-have-won, PF 0.04 — the vetoed shorts almost all failed) while LONG
vetoes graded poorly (n=43, 55.8%, PF 3.34 — the vetoed longs mostly worked),
concentrated in `too_extended_from_base` LONG (n=30, 63.3%, PF 4.42). Reasons
this cannot be read as confirmation: one matured horizon of ten; n below or
near 30 per cohort; the two machine-written caveats travel with the data (the
LIKE control's bounded picklist, and "Veto D1 — but M5 today" writing an
ordinary veto row so day-traded names count as vetoed — which biases the LONG
cohort *toward* looking wrongly-vetoed); no declared-then-frozen window; and
the CSV does not yet carry the ground-rule-10 statistics contract (no median,
trimmed mean, p10/p90, bootstrap interval — see §4). It is a hypothesis worth
the two-week gate, nothing more.

**Refuted hypotheses:**
- *Window contention* — REFUTED. Worst real night: 72.4 min of an 8-hour
  window (01:00–09:00 ET, `local_settings.json`), consistent with the plan §2
  projection of ~3.5% worst-case utilisation.
- *Silent weekend early-exit* — REFUTED by reproduction (§2 above).
- *Model/tier misallocation* — REFUTED. Slots 2–3 resolve
  `ai_local_model_medium` = `gemma3:12b-tbv3ctx` (`ai_summary.py:67-71`,
  `briefs.py:115, 873`) — the derived model with explicit `num_ctx`, exactly
  what §6.1 mandates; the stock-context defect that killed six nights of briefs
  is fixed. The large tier is RETIRED (twice falsified on this hardware, plan
  §2), its setting dormant by design; no caller of `local_model("large")`
  exists in the slate. "27b configured but unused" in the brief understates
  this: it was never usably configured — stock `gemma3:27b` never loaded at
  all.
- *The absent Daily Digest Ledger as the binding constraint* — REFUTED **for
  now**. `digests/`, `retros/`, `models/` empty is the designed state: Phase 2
  is "TO BE REDESIGNED … Do not build" pending trader sign-off on the §6.4a
  packet, and the only tier authorized to read digests (frontier) is itself
  unauthorized (§7.3). A missing substrate with no authorized reader starves
  nobody today. It becomes real the day a synthesis pass is authorized.

**UNKNOWN:** whether brief/summary content quality is *also* a constraint —
unread and unreadable are different failures and only the first is measured.
Settled by the §8 readership question, not by more code.

---

## 4. Target state (Q3)

**Design principle, taken from the evidence rather than resisted:** the layer's
proven value is deterministic grading over trader-generated streams; inference
earns a slot only where a named reader exists and the numbers are computed by
code first (the §3.2 doctrine — "numbers are computed by code, never by the
model" — which the veto slot already embodies with zero model calls). The slate
should therefore grow deterministic graders in R10's packet order and hold the
inference surface flat or smaller until readership is proven.

Sequenced against R10's actual order (later phases append, never reorder):

1. **Now, before the R10.A canary** — repair the substrate, no new slots:
   restore the Questrade chain (trader action + AI-P4 surface), land the
   nightly-status honesty packet (AI-P3) so R7 gate 6 ("≥5 consecutive nightly
   `journal_import` ledger entries with coverage advancing") becomes reachable,
   and fix the stale machine-written caveat (AI-P5). Deterministic: all of it.
2. **R10.A canary + two-week collection clock** — the checkpoint's own next
   action (`outcome_sweep_autorun="on"`, trader flip). Nothing for the AI slate
   to do except not get in the way; note the clock gating R10.I **has not
   started**.
3. **R10.F — `like_cohort_grading`**: a fifth deterministic slot appended after
   `veto_cohort_grading`, mirroring the veto trio
   (`like_cohort_{picks,outcomes,performance}.csv`). No model. This doubles the
   graded-judgement surface for the cost of seconds, and R9.2's required "why"
   note means every LIKE row now carries text a *future* opt-in scope can read.
   Not two-week-gated; buildable in packet order.
4. **R10.G / R10.H** — deterministic stores (market context ledger, market
   journal + surfaces). The journal-entry free text reaches an AI scope
   **opt-in only** (trader decision already recorded in plan.md L1118-1126).
5. **R10.C discipline applied to the cohort CSVs.** Ground rule 11 names
   "cohort performance CSVs" as evidence-facing surfaces; the current
   `veto_cohort_performance.csv` carries win rate / avg / PF only — no median,
   trimmed mean, p10/p90, bootstrap, or discovery/confirmation label. When
   R10.C builds `scripts/evidence_stats.py`, the veto (and future LIKE)
   performance rollups should be regenerated through it rather than growing a
   private second stats implementation. Deterministic.
6. **R10.I — after two weeks of R10.A/B collection**: the `evidence_report`
   slot (deterministic, no model) appended to the runner, plus the opt-in
   `market_journal` scope. Earliest honest start is ~two weeks after **both**
   the R10.A autorun canary and R10.B collection are live, and R10.B is
   unstarted — so this is last by construction.
7. **The weekly synthesis pass — design only, not authorized (§7.3).** Gate:
   two weeks of graded rows (first grading 2026-08-20 → earliest eligibility
   ~2026-09-03), plus explicit trader authorization for any frontier call.
   Recommended shape when its day comes: a deterministic `evidence_stats`
   rollup section first (free, always runs), with the model narration as a
   second section over that fact pack only — the §6.4a D1 fact/narration split
   applied to the cohort instead of building Phase 2 first. The medium tier can
   pilot it before any frontier spend is requested.
8. **Inference slots 2–3**: hold or shrink pending the §8 readership answer.
   No new nightly model reads of any raw stream (§7.3 stands).

What stays deterministic forever: every number in every slot output (ground
rule 6). What genuinely needs inference: only narration around code-computed
facts, and only once someone demonstrably reads it.

---

## 5. Packet proposals (Q4)

All are commit-sized, all leave detectors/scores/gates/alerts untouched, and
none touches a file housing detector/scoring/alert code — so the file-scoped
ask-first rule does not fire on any of them. Two carry a different flag:
AI-P2 is a **spec amendment** requiring trader assent, and all remain subject
to R10 ground rule 1's byte-identical golden fixtures.

**AI-P1 — Focus Pick Review mirror-cohort join (the R8 §6 DEFERRED item).**
Files: `scripts/ui/panels/weekend_prep_panel.py`, new test beside the existing
panel tests. Read `veto_cohort_performance.csv` through the existing
`canonical_veto_cohort` pooling; render the cohort table beside the picks↔
outcomes join that landed 2026-08-18. Honesty rules already established there
apply: an unmatured horizon is blank never 0.00%, missing CSV shows an explicit
absent state. **Bonus defect this fixes:** the pane's subtitle at
`weekend_prep_panel.py:182` already promises "the veto cohort beside them"
while nothing loads any `veto_cohort_*` file — presentation currently
overclaims. Ask-first: no. Tests: offline fixture CSV; pooling; blank-not-zero;
absent-file state. Exit gate: subtitle truthful, cohort visible. Advances: R8
§10 live observation on the Focus review step, and gives the §3 discovery
finding its designed weekend surface.

**AI-P2 — auto-tag backlog toggle (spec amendment — trader assent needed).**
R8 locked the journal hook to "the weekly auto-tag review only", and
`week_tag_candidates` scopes to the week's trades — but the backlog is 220
candidates spanning history, so the confirmation stream can only fill at the
weekly trickle. Proposal: an "all pending" toggle on the existing sub-pane
(files: `weekend_prep_panel.py`, `scripts/ui/services/journal_feed.py`),
default off, same confirm→`save_annotation` / correct→`record_tag_corrections`
paths. Ask-first: no (no detector code), but it widens an R8 locked decision,
so it needs the trader's written ok first. Tests: scoping/ordering of the new
query; existing confirm/correct paths unchanged. Exit gate: backlog
burn-downable in one sitting. Advances: R8 §10 observation 6, and directly
attacks binding-constraint hole 2.

**AI-P3 — nightly journal status honesty.** A run that imports successfully
but finds reconcile mismatches records FAILED and burns all 3 attempts
(R10.0's diagnosis; 9 lifetime failures, 0 ok). Separate the semantics:
import success with reconcile findings → `ok` with the mismatch count in the
reason (precedent: R7 §6's "zero-execution night = ok"); reserve `failed` for
runs that did not import. Files: `scripts/journal_runner.py` + its tests.
Ask-first: no; governed by R7 §6 — cite it in the commit. Tests:
mismatch-after-successful-import records ok-with-findings; true import failure
still fails; attempt cap unburned by findings. Exit gate: first honest `ok`
row. Advances: R7 §10 gates 3 and 6.

**AI-P4 — Questrade chain health surface.** A warning in the Journal Health
tab (and/or `operations_audit.py` → System Health) when
`journal_questrade_expires_at` is stale or the last Questrade coverage failure
is an oauth error, with text telling the trader to paste a fresh token into the
existing field. Files: `scripts/ui/panels/journal/health_tab.py` or
`scripts/operations_audit.py`, + tests. Ask-first: no. Exit gate: a dead chain
is visible on the desk within one session instead of discovered by audit.
Advances: R7 §10 gate 1 (full Questrade coverage backfill becomes reachable
and its regressions visible).

**AI-P5 — stale machine-written caveat.** The `trader_judgement` scope emits
"the like+claim control currently offers only the 'Main swing' group" as
code-owned caveat data; the picklist expanded 2026-08-21 (Main swing + three
post-earnings families + `second_dev_breakout`). A caveat that is wrong defeats
its own purpose. Files: `scripts/ai_summary.py` (caveat text/derivation —
ideally derive it from `capture_rail`'s claim groups so it cannot go stale
again), + test. Also reconcile the two stale doc lines found en route:
LOCAL_AI_AUTOMATION_PLAN §7.2's caveat text, and §6.4c still reading "QUEUED —
do not build yet" for a `journal_import` slot that ships first in
`default_slots()`. Ask-first: no. Exit gate: caveat matches the live picklist;
test pins it.

**AI-P6 — R10.F `like_cohort_grading` slot** (when its packet turn comes; listed
for completeness, spec already exists in plan.md L1279): mirror trio, slot
appended after `veto_cohort_grading`, UTC + `session_date` stamps.

---

## 6. Stop-doing list (Q5)

1. **Park `ticker_briefs` pending the readership answer** (§8 Q1). It is the
   entire model bill — 72 min, 65 calls, 20.55 MB of store — with unmeasured
   post-TB-5 content quality and no evidenced reader. Mechanism if parked:
   scope/slot skip, not deletion — the TB-0…TB-6 hardening is real engineering
   worth keeping. Recommended default: demote to opt-in
   (`run_ai_jobs.py`-style on-demand) until the trader says they read it.
2. **Stop burning 3 nightly attempts on structurally-failing journal runs** —
   AI-P3. Until it lands, the nightly ledger's `failed` rows are noise, not
   signal.
3. **Do not build Phase 2 narration, and do not sign off §6.4a's narrator half
   now.** The deterministic fact-pack idea survives — but as R10.C's
   `evidence_stats` + R10.I's `evidence_report`, which are authorized and
   sequenced. Building a digest narrator before any authorized reader exists
   repeats the briefs pattern.
4. **Stop watching the AI ledger for weekend rows** — silence is the designed
   behaviour (§2). The per-firing wrapper logs are the liveness signal.
5. **No frontier calls** — unchanged; nothing measured here justifies one, and
   §7.3 forbids it anyway. (The dormant large-model setting key costs nothing;
   keep.)

---

## 7. Open decisions for the trader

1. **Do you read the morning `ai_summary` and `ticker_briefs` files?** The two
   model slots' PROVEN/REFUTED status turns entirely on this; no artifact can
   answer it. *Recommended default:* park `ticker_briefs` (opt-in), keep
   `ai_summary` one more week, revisit both at R10.I. Cost of parking: a cold
   morning file if you did read it (say so and it stays). Cost of keeping
   unread: ~72 min of window and the whole model bill for nothing nightly.
2. **Questrade token reset + weekly routine.** The stored chain is dead; only
   you can mint a new refresh token in the Questrade portal. *Recommended
   default:* paste the new token into Journal ▸ Health ▸ "Questrade refresh
   token" each week (details below §7a); AI-P4 then keeps staleness visible.
   Cost of not doing it: two accounts (one TFSA) permanently absent from every
   journal analytic, and MISMATCH noise every night.
3. **AI-P2's spec amendment** — widen the R8 auto-tag review from
   "this week's trades" to an optional all-pending backlog view? *Recommended
   default:* yes, default-off toggle. Cost of no: the 220-candidate backlog
   drains at ~a-week-of-trades per weekend.
4. **Flip `outcome_sweep_autorun="on"`** (already the checkpoint's stated next
   action, decision yours). Relevance here: R10.I's two-week clock cannot start
   until R10.A collection is live. *Recommended default:* flip on the next live
   weekday, per the 2026-08-23 safety work.
5. **Weekly synthesis pass:** not decidable yet — gate is two weeks of graded
   rows (earliest ~2026-09-03) *and* your explicit authorization for any
   frontier call. When it comes due, §4 item 7's deterministic-first shape is
   the recommendation; medium-tier pilot before frontier spend.
6. **Account U4867396** appears in `journal_trader_tax_statuses` but has no
   `accounts` row (likely the unfunded IBKR TFSA). Expected, or a seeding gap?
   *Recommended default:* leave as-is; it costs nothing until that account has
   executions.

### 7a. Where to put new Questrade tokens (weekly)

No new storage is needed — the slot already exists and the nightly import
recovers automatically once it's filled:

- **GUI (recommended):** Trading Desk ▸ Journal ▸ **Health** tab ▸ the
  password-style **"Questrade refresh token"** field
  (`scripts/ui/panels/journal/health_tab.py:107-108`; saved at `:296-298`).
  Paste the new token, save. The placeholder shows the masked current token.
- **File (equivalent):** the `journal_questrade_refresh_token` key in
  `C:\Users\Aaron\AppData\Local\TradingBotV3\local_settings.json`.
- What happens next: `QuestradeImporter` re-reads the setting on every access
  (no caching, `journal_importers.py:331-363`), the next night's
  `journal_import` firing gets a fresh 3-attempt budget for the new session,
  and the code persists each rotated token (`journal_importers.py:401-402`) —
  so in principle one paste re-anchors the chain and nightly use keeps it
  alive; the weekly paste is insurance against another fork. Do **not** set the
  `QUESTRADE_REFRESH_TOKEN` env var — local settings win and the env var is a
  first-boot seed only (R7 fix 10).
- **Handling note:** `local_settings.json` also holds an OpenAI key and ntfy
  tokens in plaintext. Treat the file as secret-bearing; never commit or sync
  it.

---

## 8. What I could not measure

- **Readership** of the morning summary and briefs files — no artifact records
  it; only the trader can answer (§7 Q1).
- **Post-TB-5 brief content quality** — TB-5's 96.2%-roster measurement predates
  the fix; no equivalent measurement exists for current output.
- **How much of the 19-mismatch reconcile storm is the Questrade hole** — with
  0 Questrade days covered, missing executions and genuine mismatches are
  indistinguishable until the chain is restored.
- **Veto cohort at 3/5/10-session horizons** — not matured; only
  `horizon_sessions=1` exists in the CSV. Every §3 cohort claim is `discovery`.
- **The LIKE cohort** — R10.F not built; the trader's positive judgements are
  currently ungraded, so the veto finding has no mirror to be checked against.
- **Whether the dead Questrade chain forked from a second consumer or from
  expiry** — the ×56 500-errors carry an older token than the one stored, which
  proves the chain broke, not why. R7's "token race stated-not-solved" note
  stands. Evidence that would settle it: timestamps of the first 500 vs the
  last successful save of `journal_questrade_refresh_token`.

Missing data is uncertainty, never confirmation: an unmeasured reader is not a
valuable slot, and an unmeasured brief is not a good one.
