# Local AI & Automation Plan

Document role: **active implementation specification**, subordinate to the root
`plan.md`. Phase status is summarized in `CHANGELOG.md`; this file retains the exact
gates and design detail. Current roadmap locations: P0.6, P3.3, and P6.3.

Historical key: accepted into the former plan.md Section 12 as item 13b (trader-directed,
2026-08-08). **Phase 0 COMPLETE on branch `local-ai-phase-0` (2026-08-08):
code landed, Ollama installed and benchmarked on the main desk, all three
tiers chosen and verified, and the exit gate verified end to end. Phase 1 is
implementation-complete on `testing-week-2026-08-10` (2026-08-09); its live
five-session exit gate has not started. Phase 2 remains stopped for redesign.**
Subordinate to `plan.md` — this
document never overrides plan.md sections 5-7 or the section 12 execution
order. Section 6 is the binding implementation spec; phases execute in order,
each on its own branch.

## 1. Mission

Use the always-on Ryzen main desk (8845HS) as a **local, free-token, overnight
AI batch layer** that automates the advisory work the trader currently does by hand or
skips: AI summaries, journal enrichment, review-policy curation, and a daily
distillation of the desk's evidence into small AI-readable digests that a
frontier model can periodically synthesize. The goal is that during market
hours the trader does nothing but chart review and analysis; everything
summarizable, taggable, or draftable happens automatically off-hours.

**Product boundary (restating plan.md sec 5 — non-negotiable):**

- Everything in this plan is **one-way advisory**. No local-model output ever
  feeds a detector, a score, an alert decision, or a state machine. Same
  shadow rule as `market_state` / `greatness_monitor`; promotion of any output
  into a live decision path requires the plan.md sec 7 ladder.
- `review_policy.json` ranks and annotates only; this plan never adds a
  suppression field.
- Order execution remains permanently out of scope.

## 2. Hardware and resource envelope

| Asset | Facts | Role |
|---|---|---|
| Ryzen 7 8845HS mini-PC | 32GB DDR5 (28.8 GiB usable), Radeon 780M iGPU (RDNA3/gfx1103, UMA — measured 17.4 GiB Vulkan heap; **Vulkan backend, not ROCm** — see Phase 0 finding 1), always-on | **Main Trading Desk** + LLM inference host + automation scheduler |
| Former desk (i5-8600K/32GB, RTX 3080 Ti) | Powered down most days — power draw, office heat, and workstation/gaming tax separation | Discord/chat box; possible ad-hoc alternative scanner. Holds **no always-on or writer role** in this plan |
| File server | 10TB now, expandable to 100TB+ | AI store (digests, model files, bulk outputs) + candidate `research_store_dir` home |
| Cloud frontier model | Fable 5 / current best, metered | Periodic synthesis passes only — reads digests, never raw bulk data |

Machine roles resolved 2026-08-08: the 8845HS **is** the main desk. Every
role this plan assigns — inference host, scheduler, `ai_store` writer — lives
on the main desk, alongside the journal, the DAS lake writer, and the ntfy
push origin it already owns. One always-on box owns everything; the 3080 Ti
is not part of the topology because its box is usually off.

**The market-hours constraint is a hard rule, not a preference.** During
market hours the main desk runs the full trading complement: only ~30% of RAM
(~10GB) and 30-50% of CPU are free. Therefore:

- **Market hours (pre-open through close): no local inference at all** by
  default. Jobs queue; nothing runs. A single optional exception — a ≤4B
  model, one request at a time, `keep_alive=0` so it unloads between calls —
  may be enabled later per-job, but the plan ships with the window closed.
- **Off-hours window (evening after post-session artifacts land → pre-market
  prep):** the full budget. 12B-27B quantized models are the workhorses here;
  tokens/sec is irrelevant at 2am.
- The inference server is configured to never hold memory idle:
  one model loaded at a time, `num_parallel=1`, aggressive unload.

Model tiering:

| Tier | Example | Use |
|---|---|---|
| Small (~4B Q4) | Gemma 3 4B | high-volume classification, tagging, extraction — jobs with many small calls |
| Medium (12-14B Q4) | **`gemma3:12b-tbv3ctx`** (derived, `num_ctx 12288`) | nightly digests, journal summaries, briefs — the workhorse tier |
| ~~Large (local)~~ | **RETIRED 2026-08-10** | see below |
| Frontier (cloud, metered) | Fable 5 / best available | periodic synthesis over digests **plus** review-policy drafting and weekly retros |

**The local large tier is retired.** It was specified as "27B+ Q4, ~18GB via UMA"
and has now been falsified twice on this hardware: `gemma3:27b` never fit the
17.4 GiB Vulkan heap (Phase 0 finding 2), and its Q3_K_M replacement — verified
loading in Phase 0 — no longer loads at all beside the running desk, failing with
`alloc_tensor_range: failed to allocate Vulkan0 buffer` at ~1 GB contiguous
allocations on a contended UMA heap. A tier that only works when the trading desk
is closed is not a tier on an always-on desk. Its jobs (review-policy drafting,
weekly retros) move to the frontier row, which is cheap here precisely because
this plan only ever sends it small digests. The `ai_local_model_large` setting key
stays, dormant and free to keep, so a future revisit is a settings change.

*Revisit triggers for a local large tier:* Ollama Vulkan allocator improvements,
working ROCm on gfx1103, or a RAM upgrade. Until one lands, assume no local large
model — Phase 2's redesign and Phase 4 both depend on this.

## 3. Architecture

### 3.1 Inference endpoint (the only new plumbing)

Ollama (or llama.cpp server) on the main desk exposing an OpenAI-compatible
endpoint on localhost (the desk is single-machine; satellites are retired). Both existing AI call sites already speak this shape:

- `market_prep/services/ai_service.py` constructs `OpenAI(api_key=...)` — add
  a configurable `base_url` setting.
- `scripts/ai_summary.py` selects among hardcoded provider URLs
  (`OPENAI_RESPONSES_URL`, `ANTHROPIC_MESSAGES_URL`) — add a third `"local"`
  provider with a configurable URL and model name.

That is the entire integration surface. Every existing AI feature then runs
against local models by flipping a setting, with cloud providers unchanged as
an option per call site.

### 3.2 The Daily Digest Ledger (backbone)

The central design, and the answer to "throw big files at a small model?" —
**never**. Small models degrade on long contexts and hallucinate under
information overload. Instead:

- Every evening, a digest job distills **that day only** into one small,
  schema-versioned JSON file (target: a few KB, hard cap well under the
  medium model's context).
- **Numbers are computed by code, never by the model.** Win rates, R
  multiples, conversion stats, scan counts come from deterministic pandas /
  `review_learning.py` aggregation. The LLM only writes the narrative
  fields, tags, and "what stood out" annotations around facts it is handed.
  A model that cannot do arithmetic reliably is never asked to.
- Every digest entry carries **evidence pointers** (source file + record IDs
  / timestamps) so a later reader — human or frontier model — can drill from
  a claim down to raw evidence.
- Digests are **point-in-time**: built only from information available that
  day, timestamps with explicit timezones (plan.md sec 5 research rule),
  append-only, never rewritten. A bad digest gets a superseding sibling, not
  an edit.

Daily inputs (all already exist):

| Source | Path | Content |
|---|---|---|
| Alert review decisions | `alert_review_events.jsonl` (+ per-day dir) | what fired, what the trader did |
| Review scoreboard | `review_learning.py` state | revealed-preference aggregates |
| Journal | `trade_journal.sqlite3` (opportunity events: SEEN/TAKEN/SKIPPED/…) | trades and notes |
| Scan evidence | run manifests, `job_ledger.jsonl`, shadow JSONLs under `%LOCALAPPDATA%\TradingBotV3\diagnostics\` | what the engines saw |
| Auto/Away digest | `autopilot_today.txt` (read-only) | what the phone was told |

Then the two-stage pattern the trader described, which is standard
hierarchical (map-reduce) summarization and is **realistic**:

1. **Map (nightly, local, free):** day → digest. 365 days ≈ 365 small files.
2. **Reduce (weekly/monthly, frontier, cheap because inputs are small):**
   Fable-5-class model reads the last N digests — not raw data — to find the
   winners: which setups, conditions, and habits are working. Output is a
   synthesis report with drill-down pointers. Reading 30-90 digests of a few
   KB each is a trivial context load for a frontier model; reading the raw
   JSONL/Parquet behind them would be impossible. The digest schema is
   designed for this reader from day one.

### 3.3 Storage: the AI store

A new `ai_store` root on the file server, **main desk as sole writer**
(plan.md sec 5: one component owns each mutable shared export):

```
ai_store/
  digests/YYYY/YYYY-MM-DD.json      # daily digest ledger (append-only)
  briefs/                           # per-ticker/morning briefs
  retros/                           # weekly/monthly retro + synthesis reports
  models/                           # GGUF files (large, no reason to redownload)
  logs/ai_job_ledger.jsonl          # every job: model, tokens, duration, exit
```

- **Not in the `C:\TradingBotData` home folder** (that folder is the compact
  operational storage class, and the hourly cold push mirrors its subtrees
  wholesale; it was Drive-synced until decision 0015) and
  **not inside the DAS lake tables** — the lake and the AI store are separate
  storage classes with separate writer components. Both components now live
  on the same main desk, which is fine: ownership is per-component, not
  per-machine, and keeping the trees separate means an AI-job bug can never
  corrupt lake data.
- Small human-facing outputs only (morning brief, weekly retro) additionally
  publish to the home folder via the existing atomic-publish pattern:
  temp file → verify → rename; a failed publish never destroys the last
  verified copy.
- `autopilot_today.txt` is untouched — it keeps its single verified writer.

### 3.4 Scheduling

Jobs run on the main desk; the `master_avwap_mini_pc.py` slot/state pattern
is the scheduling template (named slots, per-slot status,
skip-don't-pile-up on overrun). Every job writes a ledger
row. Job failures degrade to "no digest tonight," never to a corrupted
artifact — same failure philosophy as the report writers.

## 4. Phases

Ordered; each phase is independently shippable and independently useful.
Trader priorities: Phases 1, 3, 4 are the stated wants; Phase 2 is the
foundation 4-6 stand on.

### Phase 0 — Endpoint + plumbing (no product behavior change)

- Install Ollama on the main desk; pull one model per tier; benchmark tok/s and
  RAM footprint for each on this exact box, recorded in this doc.
- Add the `"local"` provider to `ai_summary.py` and `base_url` support to
  `ai_service.py`, config-gated, defaulting to current behavior. Tests for
  provider selection.
- Exit gate: existing AI summary produces sane output against the local
  medium model; no test regressions; cloud path unchanged.

**Status 2026-08-08 — code half DONE, operator half PENDING** (branch
`local-ai-phase-0`; 1856 passed, 7 subtests; smoke 7/7):

- `scripts/ai_summary.py` gained the `local` provider
  (`local_endpoint_url` / `local_provider_enabled` / `local_model` /
  `default_model_for`), posting the OpenAI **chat-completions** shape to
  `{ai_local_endpoint_url}/chat/completions` with the placeholder key, one
  retry on invalid JSON, and the same `validate_ai_summary` evidence checking
  the cloud providers get.
- `market_prep/services/ai_service.py` gained `base_url` support. One setting
  (`ai_local_endpoint_url`) flips both call sites; either can be pinned back to
  a cloud URL through its own `market_prep_ai.base_url`. **Deviation from sec
  6.2, deliberate:** a base-URL deployment also switches from
  `client.responses.create` to `client.chat.completions.create`, because Ollama
  and llama.cpp implement chat-completions and not the Responses API — passing
  `base_url` alone would have produced a 404 on every call. The cloud path
  still uses the Responses API unchanged.
- The A.I. workspace panel lists "Local (on this desk)" only when
  `ai_local_endpoint_url` is set, so an unconfigured desk sees exactly the two
  providers it always saw.
- `tests/test_local_ai_provider.py` asserts the negative case that matters:
  with the new settings unset, both cloud providers receive a byte-identical
  URL, JSON payload and headers.

**Exit gate MET for the small and medium tiers (2026-08-08).** Ollama 0.32.6
installed on the main desk via winget, all three tiers pulled, endpoint
verified, `ai_local_endpoint_url` set to `http://127.0.0.1:11434/v1`, and a
real `request_ai_summary(provider="local")` run produced a schema-valid,
evidence-grounded summary that passed `validate_ai_summary` and exported
cleanly (65.3 s end to end on `gemma3:12b`).

#### Benchmark, measured on this 8845HS (28.8 GiB usable RAM)

| Tier | Model tag | Device | Footprint | Gen tok/s | Prompt tok/s | Cold load |
|---|---|---|---|---|---|---|
| Small | `gemma3:4b` | 100% iGPU | 2.9 GB | 24.4 | 144.5 | 12.5 s |
| Medium | `gemma3:12b` | 100% iGPU | 8.1 GB | 8.8 | 94.5 | 16.9 s |
| Large | `hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M` | 100% iGPU | 14 GB | 4.1 | 26.9 | 184.5 s |
| *(rejected)* | `gemma3:27b` (Q4_K_M, 17 GB) | — | — | **does not load** | — | — |

Method: Ollama's own `eval_count` / `eval_duration` counters (exact token
counts, nanosecond durations) over a digest-shaped prompt, not wall-clock
guessing; footprint from `ollama ps`. Reference point for the iGPU's value:
`gemma3:12b` on CPU only managed 6.6 gen / 37.0 prompt tok/s.

Projected job runtimes from the measured large-tier rates, worst case:
nightly policy draft (20k in / 3k out) ≈ 12.4 min prefill + 12.2 min
generation + 3.1 min cold load ≈ **28 min**; weekly retro (30k in / 4k out)
≈ 18.6 + 16.3 + 3.1 ≈ **38 min**. Against the 13.5-hour off-hours window that
is ~3.5% utilisation on the worst night, so the window is not a constraint on
any tier.

#### Two hardware findings that change the sec 2 assumptions

1. **The 780M needs `OLLAMA_IGPU_ENABLE=1`, and does *not* go through ROCm.**
   Ollama drops the ROCm device outright — `no rocblas support for gfx target
   gfx1103` (its bundled rocblas covers gfx1030/1100/1101/1102/1150/1151/
   1200/1201/906, not the 780M's gfx1103). The working path is Ollama's
   **Vulkan** backend, which detects the 780M but skips it by default because
   it is integrated. With the flag set, both shipped tiers run 100% GPU. The
   `HSA_OVERRIDE_GFX_VERSION` hint in Ollama's own log was **not** used: it
   maps gfx1103 onto kernels built for another target, and silently-wrong
   numerics are a bad trade for an evidence pipeline.
2. **The stock 27B does not fit; a Q3 quant of it does — RESOLVED.** The
   Vulkan iGPU heap is 17.4 GiB total / 16.5 GiB available. `gemma3:27b`
   (Q4_K_M, ~15.8 GiB of weights) loads its weights and then fails allocating
   its ~210 MB compute buffers:
   `ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer`. Reducing
   the context to 2048 did not rescue it — a ~210 MB miss.

   Decision (frontier-model review, 2026-08-08): enter Phase 4's two-week gate
   with **the highest-ceiling candidate that actually loads**, since a model
   swap is a free settings change and the gate — not speculation about
   quantization damage — is the real quality arbiter. Parameter count tends to
   matter more than a modest quant step for judgment-heavy prose, and speed is
   irrelevant at 2am, so the 14B option's only advantage evaporates.

   Ollama's own library has **no** Q3 build of gemma3:27b (its 27b tags are
   Q4_K_M 17 GB, QAT 18 GB, Q8_0 30 GB, FP16 55 GB — all too large), so the
   model is sourced through Ollama's supported Hugging Face route:
   **`hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M`**, a 12.51 GiB file
   that leaves ~4 GiB of heap for compute buffers and KV cache. Verified: it
   loads 100% on the iGPU and produced a schema-valid, fully-cited summary
   through the real `request_ai_summary` path (6/6 sections, 20 evidence
   citations, 218.9 s wall including cold load).

   Not chosen, and why: `Q4_K_S` (14.60 GiB) also exists and is a higher
   quant, but leaves only ~1.9 GiB of margin — below the comfort the Q4_K_M
   failure argues for. It is the obvious step up if Q3 disappoints in the gate.

   **Revisit triggers:** if the Q3-27B drafts materially lose Phase 4's
   two-week side-by-side against the cloud model, switch the setting to a
   14B-class model and rerun the gate; if that loses too, stay on cloud, which
   the plan already blesses. Independently, retest `gemma3:27b` Q4_K_M if a
   future Ollama/llama.cpp release improves Vulkan compute-buffer allocation
   or ROCm gains gfx1103 — it missed by ~210 MB.

   **Operational note for Phase 4:** the large tier's cold load is 184.5 s
   against an `OLLAMA_KEEP_ALIVE` of 60 s, so a job whose calls are more than
   a minute apart pays three minutes of reload each time. Large-tier jobs must
   either batch their calls or pass a longer per-request `keep_alive`.

#### Server configuration applied (sec 2: never hold memory idle)

Set as **user-level environment variables** on the main desk, so the tray app
picks them up at login:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_IGPU_ENABLE` | `1` | finding 1 — without it everything runs on CPU |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | sec 2: one model loaded at a time |
| `OLLAMA_NUM_PARALLEL` | `1` | sec 2 |
| `OLLAMA_KEEP_ALIVE` | `60s` | sec 2 aggressive unload, while still letting one job's burst of calls reuse a loaded model |

Ollama autostarts from a **Startup-folder shortcut**, i.e. at user login rather
than at boot — the same assumption the 07:00 launch task already makes (it also
runs in the logged-on session). Worth knowing before Phase 1 schedules
unattended overnight jobs.

### Phase 1 — Automated AI summary (trader priority #1)

- Schedule the existing advisory AI summary to run unattended off-hours
  against the local endpoint — no more manual triggering.
- Extend to **per-ticker briefs** for the Focus list / watchlists (previously
  uneconomical against metered APIs; free locally). Published to
  `ai_store/briefs/` + small morning file to the home folder.
- Exit gate: **superseded 2026-08-09 — see "Phase 1 exit gate" below.** The
  original wording ("a week of mornings where the summary and briefs are
  waiting before pre-market prep with zero manual action") defined neither a
  session nor "clean", and set no reset condition — which is how three
  Saturday artifacts came to look like coverage.

#### Phase 1 exit gate (Sol 5.6 verification review, 2026-08-09)

> Five consecutive clean NYSE sessions.

A session counts toward the five only if **all** of the following hold:

1. The day is a **regular NYSE trading session** by `market_calendar`.
   Weekends and holidays are not sessions and never count — neither toward
   the five nor as a break in them. The run is over *sessions*, not over
   calendar days.
2. The ledger carries a **canonical `ok` row keyed to that session date**,
   written by the scheduled task. `manual_test` never counts: an operator run
   publishes real artifacts but is not that session's nightly brief.
   `degraded_no_narrative` never counts. `failed` never counts.
3. **Session attribution is correct** — artifacts and ledger row are keyed to
   the session whose evidence was read, not to the wall-clock date of the run.
4. **Coverage reconciles**: the usable and excluded counts in the published
   document match the evidence package, and every excluded source carries a
   status and a reason.
5. No `correction` row retracts that session's coverage.
6. **The canonical set is complete by 09:00 ET** the next morning. Failed or
   degraded attempts during the night are tolerated **only if** the ledger
   shows automatic recovery to a canonical `ok` before 09:00 with no
   operator intervention (Sol's rule — a transient that self-heals is the
   repetition design working, not a failure of the week).
7. **No market-hours inference**: no model call during RTH, and the model is
   unloaded by 09:00 ET (spot-check `ollama ps` at least twice in the week).
8. **No duplicate canonical output**: repeated 30-minute firings never
   produce a second canonical set for the same session.
9. **Journal import health is stated** in every published document's
   data-quality section (lag, newest execution) — stalled is acceptable,
   silent is not.
10. **No provenance fallback**: the tracker vintage reads from
    `data_session` all week; any fall-through to the `updated_at` heuristic
    on a current-format payload is a defect observation.

**Preconditions and one-time drills** (each performed once, any day inside
the measurement week; the week cannot PASS without them, per the Sol 5.6
verification review):

- **Pinned checkout**: the production tree sits on one reviewed `main`
  commit, clean, for the entire week. Any code change — committed or
  uncommitted — in the production tree restarts the observation.
- **Frozen-exe guard drill**: fire the desk launch task while a desk is
  running (frozen-exe variant included); exactly one desk process, one IB
  client set, one writer remains.
- **Mid-session restart drill** (planned, on a quiet session): desk killed
  and auto-relaunched; regime collection audit ends HEALTHY with nonzero
  backfill, correct `capture_mode` and `data_session`, no reconstructed
  frozen snapshots, and no scoring/config hash change.
- **NAS transient**: one NAS unavailable/wake event observed or simulated;
  the next firing recovers, the prior verified artifact is untouched, and
  no manifest advertises a partial set.

**Reset conditions.** The count returns to **zero**, not to four, on any of:

- a session with no canonical `ok` by 09:00 ET (including one covered only
  by `manual_test`, or where failed/degraded attempts did not auto-recover);
- a session whose attribution or coverage is later found wrong — that is,
  any `correction` row appended against it;
- a pseudo-session artifact (weekend/holiday keyed as a session);
- a wrong-date narrative (stale content described as the target session);
- any model inference during market hours;
- a duplicate canonical artifact, or a duplicate desk instance;
- any automatic scoring/config mutation (tuner, calibration, config hash);
- a provenance fallback on a current-format tracker payload;
- a permanent data gap finalized without its full retry entitlement;
- a published citation of a non-usable source;
- an incomplete NAS publication (partial artifact set advertised);
- a manual corrective intervention of any kind;
- an UNHEALTHY regime collection audit for that session;
- any change to session identity, evidence packaging, the validator, or the
  failure policy. Changing the thing being observed restarts the observation.

**The manual Saturday artifacts are noncanonical.** The three 2026-08-08 `ok`
rows were written by manual runs on a day the exchange never opened. They have
been retracted by an appended `correction` row and count for nothing. The
five-session clock had not started as of 2026-08-09.

**Status 2026-08-09 — Phase 1 implementation COMPLETE; live exit gate NOT
STARTED.** The original scheduled-summary slice landed on branch
`local-ai-phase-1` (1889 passed, 7 subtests; smoke 7/7). The per-ticker and
small-morning-file completion slice landed on `testing-week-2026-08-10`. The
exit gate is **not** met: it needs five unattended session mornings, which is
elapsed desk evidence and is deliberately not started by code.

Built:

- `scripts/ai_jobs/` — headless package (core requirements only, no Qt):
  `store.py` (location, refusal, availability probe), `window.py` (the two
  gates), `ledger.py` (append-only rows + idempotency authority),
  `runner.py` (named slots, skip-don't-pile-up), `briefs.py` (the summary job).
- `scripts/run_ai_jobs.py` — the standalone program Task Scheduler boots and
  that exits. Exit codes: 0 nothing-to-do-or-succeeded, 1 a job failed, 2 the
  store was unreachable so nothing ran. `--status`, `--slot`, `--force`.
- `scripts/register_ai_jobs_task.ps1` — registers **TradingBotV3 AI Jobs**.

Verified end to end on the real desk: 18 live evidence sources, 90.4 s on
`gemma3:12b`, four files published to the NAS, exit 0, one `ok` ledger row.

Completed by the fresh trader instruction dated 2026-08-09, which supersedes
the 2026-08-08 deferral:

- `ticker_briefs` is the second named Phase 1 runner slot. It reads Focus and
  watchlist membership without modifying those files, projects the existing
  evidence package per symbol, and calls the existing provider-neutral local
  endpoint at the medium tier. The hard off-hours gate is repeated inside the
  job before every inference call.
- Full validated result/evidence/manifest packages publish below
  `ai_store/briefs/<year>/<session>/tickers/<symbol>/`. Only the bounded,
  advisory `ai_morning_brief.txt` crosses into the home folder. That
  single-writer publication is staged, byte-verified, and atomically replaced;
  a failed publication leaves the prior verified file intact.
- Neither output is imported by detector, scoring, alert, or state-machine
  modules. This completion does not implement Phase 2 and does not advance or
  initialize the Phase 1 exit-gate count.

#### Amended 2026-08-08 — evidence packaging, as built

The checkpoint review's second review found that the summary job was honest
about what it *had* and silent about what it did not, which for an unattended
nightly read is the more dangerous half. Repaired on this branch:

- **Semantic source statuses.** `available` used to mean only "the file
  exists". A source is now `available`, `empty`, `missing`, `invalid`,
  `unavailable` or `unfunded`, decided by *content*: whitespace-only text, a
  CSV with a header and no data rows, JSONL with no valid records, and JSON
  whose containers are all empty are **empty**; an unparseable document is
  **invalid**, never empty; one that cannot be read is **unavailable**.
- **A fair, priority-aware budget.** The 80,000-char package budget was
  first-come — sources were encoded in scope order and each took whatever was
  left, so one large `daily_report` could silently zero every later scope,
  including `setup_trackers` and `journal_review`, the two scopes this job
  exists to read. A zeroed source arrived with empty content and status
  `available`, indistinguishable from a genuinely empty one, and when the
  remainder hit exactly zero it did not even carry the `[package budget
  reached]` marker. The budget is now split per scope by priority weight
  (`setup_trackers` and `journal_review` 3, `daily_report` and
  `market_conditions` 2, the rest 1), a scope needing less returns its surplus
  to whoever is short, and each source that has to be shortened carries an
  in-band banner (`[showing most recent N of M rows]`, most-recent kept for
  tabular). A source that cannot be funded is **excluded and declared
  `unfunded`**, stating its real size. **A source with real bytes on disk is
  never presented as empty.**
- **The model package carries only usable sources.** Empty, missing, invalid,
  unavailable and unfunded sources go to a machine-owned `coverage` block
  (id, label, scope, status, reason) that is *not* sent to the model. The
  prompt gains one line: sources not listed were empty, missing, invalid or
  unfunded; a system data-quality note already records each one; do not
  speculate about them and do not cite them.
- **Session scoping.** `briefs.run_daily_summary` passes its `session_date`
  into packaging. Every source records the session it represents, and one
  whose artifact is from a different session is flagged stale **in band** (a
  notice the model sees) and in coverage. The 2026-07-30 `auto_report`
  incident now reads as staleness rather than silence.
- **Validator, one for every provider.** `evidence_refs` must resolve to a
  *usable* source; the rejection names the id and why it is not citable. Every
  section but `executive_summary` may be an empty array — a thin night is a
  correct answer, not a malformed one.
- **Deterministic coverage.** After validation, *code* merges provenance rows
  into `data_quality` with exact counts, prefixed `[system]`. Asking the model
  to report its own coverage produces a paraphrase of counts it cannot verify,
  and a data-quality section is the last place that belongs.
- **Failure policy.** A citation of a non-usable source fails validation; the
  retry carries the exact rejection back to the model; a second failure
  publishes a templated, model-free **DEGRADED** document stating what
  happened plus the coverage section. Zero usable sources skips the model
  entirely. The ledger gains `degraded_no_narrative`, distinct from `ok` and
  deliberately not counted as completed, so the next 30-minute firing retries
  it. Publishing nothing would have left yesterday's brief in place looking
  like a healthy night.

#### Amended 2026-08-08 — three hard-rule gaps closed

"No local inference during market hours" (sec 2) had three ways around it:

1. `window._session_bounds` returned `None` when `market_session` could not be
   imported or the calendar raised, and the block read `None` as "not a session
   day" — so a broken calendar unlocked inference for a whole trading day. It
   now raises and the block **fails closed**, treating an unanswerable day as a
   session. Weekends still short-circuit before the calendar.
2. `--force` short-circuited past the session block. It is now a *window*
   convenience only — it skips window timing and the already-done check, and
   never the session block, at either the pre-launch check or the between-jobs
   re-read.
3. `--status` called `store_available()`, which creates the store skeleton and
   writes a probe file. "Print state, run nothing" now writes nothing:
   `store_available(read_only=True)` plus `create=False` on the store
   subdirectory helpers and `ledger_path`.

#### Architecture decision: separate process, not GUI-hosted

The batch layer is its own program rather than a thread inside the Trading
Desk. Four reasons, and the first is decisive:

1. **The lifecycles are opposed.** The GUI is meant to be up during market
   hours; this layer must not run during market hours.
2. **The durability packet actively fights GUI hosting.** The 07:00 task
   relaunches the GUI every 15 minutes through the session, which would orphan
   a long job living inside it.
3. **Crash isolation.** A 14 GB model load that goes wrong must not be able to
   take down the window the trader watches charts in.
4. **It makes the sec 2 hard rule a scheduler fact**, not only a code check.

The GUI's role is a read-only view over `ai_job_ledger.jsonl` — visibility,
never ownership, so "one component owns each timer/job" still holds: the AI
runner owns AI jobs, the desk owns trading jobs, the trees are separate, and
the AI layer touches no IB client at all.

#### Scheduling shape: repeat, don't fire once

The task repeats every 30 minutes across the window rather than firing once,
and the runner asks the ledger whether each job already completed for the
session date. A healthy night no-ops on every repeat; a night where the NAS
was asleep or the endpoint was down at 01:00 self-heals at 01:30. This is the
durability packet's Tier A lesson applied to the batch layer.
`MultipleInstances IgnoreNew` is skip-don't-pile-up in scheduler form — two
runners would race the same ledger and the same endpoint.

The task runs **as the logged-on user, not SYSTEM**: SYSTEM has no network
credentials and could not reach the UNC store at all.

### Phase 2 — Daily Digest Ledger (foundation)

- Deterministic extraction layer (code, no LLM): pull the day's facts from
  the sources in 3.2 into a typed intermediate.
- Digest writer: medium model narrates/tags around the extracted facts;
  schema-versioned JSON with evidence pointers; append-only under
  `ai_store/digests/`.
- Exit gate: 10 consecutive session days of digests; trader spot-audits ≥3
  against raw evidence and finds no fabricated facts (numbers all traceable
  to the deterministic layer).

**Status 2026-08-08 — TO BE REDESIGNED, DESIGN PENDING. Do not build.**

Phase 1's repairs changed what Phase 2 should be. The load-bearing lesson is
that everything trustworthy in the nightly output came from *code* — the
coverage block, the exact counts, the status of every source — and everything
that needed guarding came from the model. So Phase 2 will be redesigned around
**deterministic fact packs**: the extraction layer becomes the product rather
than the input to a narrator, and any narration sits on top of facts that are
already complete, counted and citable.

That redesign has **not been done**. Concretely, for the next agent:

- The draft digest schema in sec 6.4 is **not** the schema to build. It was
  drafted before the fact-pack direction and has never had trader sign-off.
- **Do not build or freeze any digest schema in this session or the next one
  without a design packet first** (trader decision 2026-08-08). A schema
  written into an append-only store is expensive to take back.
- The digest-sufficiency benchmark named in the confirmation register is part
  of that pending design, not a separate task to start early.
- **A design packet now exists: sec 6.4a (PROPOSED, 2026-08-10).** It is a
  proposal awaiting trader sign-off, not authority to build. Its open questions
  must be answered first — question 1 (what counts as "winning") is a trading
  judgement no agent should make.

### Phase 3 — Journal enrichment (trader priority #2)

- Nightly pass over new journal rows: summarize, tag with setup names from
  `docs/SETUPS_MAJOR.md` / `SETUPS_TEST.md`, link entries to that day's
  alert/review evidence. Augments the existing `AutoTagger`
  (`journal_analytics.py`) — LLM tags land in advisory fields, never
  overwrite trader-entered data.
- **Journal scaffolding:** pre-fill draft entries from the day's opportunity
  events (SEEN/TAKEN/SKIPPED already in the schema) so the trader only
  annotates instead of transcribing.
- Weekly retro: large model writes a review of the week's journal + digests
  to `ai_store/retros/`.
- `trade_journal.sqlite3` is machine-local (`PERSISTENT_RUNTIME_DATA_DIR`)
  on the main desk — which is also the AI host, so enrichment jobs run
  locally against it. Writes go through the existing `JournalStore` API only
  (advisory fields), preserving the single-writer discipline.

### Phase 4 — Review-policy curation (trader priority #3)

- The `docs/REVIEW_LEARNING_LOOP.md` AI step (read review artifacts → write
  `review_policy.json` rank/annotate output) moves to **the frontier model, or
  the local medium tier**, nightly. *(Amended 2026-08-10: the local large tier
  is retired — sec 2. The existing two-week side-by-side gate below is the
  arbiter of which one earns the job; it was always a quality comparison, and
  it does not care which model is on the other side.)*
- **Validation gate before it goes live:** two weeks of local drafts written
  to `review_policy_draft.json` only, compared side-by-side against the
  cloud model's output; trader signs off on quality before the local model
  writes the live file. Quality here matters more than cost — if the 27B
  class isn't good enough, this phase stays on the cloud model and that is a
  fine outcome.
- Invariants restated: ranks and annotates only; no suppression field; FIFO
  queue ordering untouched.

### Phase 5 — Frontier synthesis pass ("find the winners")

- Weekly (and monthly) job: frontier model reads the last N digests + retros
  and writes a synthesis report — which setups/conditions/habits are
  winning, with drill-down pointers into raw evidence. Human-triggered at
  first; scheduled once trusted.
- Any synthesis claim that would change behavior (tune a threshold, favor a
  setup) routes through the normal plan.md sec 7 evidence process — the
  report is study material, not a control signal.

### Phase 6 — Deferred / optional backlog

Explicitly deprioritized (trader has external daily market briefs), plus
ideas not yet asked for. Ordered by leverage-per-effort:

1. **Morning pre-flight page:** one artifact before open — earnings flags on
   watchlist names (`earnings_history.py`), overnight gaps, System Health /
   `operations_audit.py` anomalies, yesterday's digest highlights.
2. **Nightly ops digest to phone:** `operations_audit.py` +
   `review_capture_audit.py` results summarized and pushed over the existing
   ntfy channel — the "is the machine healthy" glance without opening the desk.
3. **Setup-doc drift watch:** periodic comparison of `SETUPS_MAJOR.md` /
   `SETUPS_TEST.md` stated definitions against accumulated digest evidence;
   flags drift for trader review (docs are trader-owned; the job only flags).
4. **News sidecar, RSS-first:** the cheap version reuses the existing
   `feedparser` market-prep ingestion — small model classifies headlines for
   watchlist relevance overnight, medium model writes a morning section.
   A websearch-based scanner is a bigger build (search API, dedup, paywalls,
   rate limits) for marginal gain over RSS + the trader's existing external
   briefs — build the RSS version first if this phase activates at all.
5. **Warehouse enrichment:** once the research warehouse accumulates
   history, overnight labeling/annotation jobs over the lake — gated behind
   `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, not scheduled here.

## 5. Invariant compliance map

| plan.md sec 5 invariant | How this plan complies |
|---|---|
| Decision-support only | All outputs are documents a human reads |
| Shadow engines never influence live decisions | No AI output feeds detectors/scores/alerts; Phase 4 writes the already-advisory policy file under a validation gate |
| No detector/scoring change without golden fixtures | No detector or scoring code is touched anywhere in this plan |
| Completed bars only; missing data is uncertainty | Digests are built post-session from completed artifacts; absent evidence is recorded as absent |
| Watchlist names never auto-removed | No writer in this plan touches watchlists |
| One owner per mutable export; failed publish never destroys last verified | Main desk solely owns `ai_store`; home-folder copies use atomic publish; `autopilot_today.txt` and lake writers unchanged |
| Point-in-time research, explicit timezones | Digests are per-day, append-only, tz-explicit |
| `review_policy.json` ranks/annotates only | Restated as a Phase 4 gate; no suppression field ever |

## 6. Implementation appendix (binding for the implementing agent)

This section removes the design decisions so implementation is mechanical.
Where it conflicts with an earlier section, this section wins.

### 6.0 Ground rules

- One branch per phase (`local-ai-phase-0`, …), merged to `main` only with the
  full suite green; commit small and push after each commit (CLAUDE.md rules).
- New code lives in `scripts/ai_jobs/` (new package) plus the two provider
  call sites named in 6.2. **No file under `bounce_bot_lib/`,
  `master_avwap_lib/`, detector, scoring, or alert paths is touched in any
  phase.** If a phase seems to need it, stop and ask the trader.
- Every feature is config-gated and **default-off**: with the new settings
  unset, app behavior is byte-identical to today (tests enforce this).
- Unit tests are deterministic and offline — no live endpoint, clock injected
  (the pinned-session-clock pattern already used in `tests/`); anything
  hitting a real endpoint is marked `network`.
- All settings go through `get_local_setting` / `save_local_setting`
  (`scripts/project_paths.py`) — no new settings mechanism.
- Installing/starting Ollama is an **operator step, not code**: the agent
  ships a short runbook section in this doc, the trader installs and starts
  the service. Code must degrade gracefully (log + skip) when the endpoint is
  unreachable.

### 6.1 Config keys (all in `local_settings.json`)

| Key | Default | Meaning |
|---|---|---|
| `ai_local_endpoint_url` | unset = local provider disabled | e.g. `http://127.0.0.1:11434/v1` |
| `ai_local_model_small` | `gemma3:4b` | high-volume classification tier |
| `ai_local_model_medium` | `gemma3:12b` | digests, briefs, summaries. **Desk value: `gemma3:12b-tbv3ctx`** — see the derived-model rule below |
| `ai_local_model_large` | `gemma3:27b` | **DORMANT** since 2026-08-10: the local large tier is retired (sec 2). The key is retained so a revisit is a settings change |
| `ai_local_evidence_budget_chars` | `22000` | evidence ceiling for **local** calls only; `MAX_TOTAL_EVIDENCE_CHARS` (80,000) remains the cloud ceiling. Derivation: 12288 context − 3500 generation − ~1000 scaffold ≈ 7,800 evidence tokens × ~3.0 chars/token, rounded down so the **retry** (which re-sends the evidence plus the validator's rejection) still fits. A non-positive or unparseable value falls back to the default rather than funding nothing |
| `ai_store_dir` | unset = AI store + all jobs disabled | file-server or local path; **refuse any path inside the `C:\TradingBotData` home folder**, mirroring `research_warehouse/config.py`'s refusal. A local-disk path is fine while the file server pends — implementation never blocks on server setup |
| `ai_offhours_start` / `ai_offhours_end` | `"18:30"` / `"08:00"` | ET wall-clock (`zoneinfo`, `America/New_York`) job-launch window. Weekends: all day allowed. Holidays treated as normal weekdays (conservative — the window still applies). No job **launches** outside the window; a job that crosses the end finishes its current model call and stops gracefully |

Model tags are starting picks; the Phase 0 benchmark may swap them by editing
these settings — never by hardcoding.

**Any medium- or large-tier tag MUST be a derived model with an explicit
`num_ctx`.** A stock Ollama tag inherits the server's default context, which
measured **2,048 prompt tokens** on this desk. That default is the root cause of
the six-night `ticker_briefs` failure: 80,000 chars of evidence were sheared to
2,048 tokens, the model answered from the fragment, and generation — sharing the
same window — ran out mid-JSON, surfacing only as `Unterminated string`. Create
the tier model explicitly:

```
FROM gemma3:12b
PARAMETER num_ctx 12288
```

Per-model `num_ctx` rather than the global `OLLAMA_CONTEXT_LENGTH`: one global
value cannot serve both tiers here. 16384 fails to allocate outright, and a value
large enough for briefs starves everything else on a shared UMA heap.

### 6.2 Provider plumbing spec (Phase 0)

- `scripts/ai_summary.py`: add `"local"` to `DEFAULT_MODELS` (default =
  the medium-tier setting) and teach `normalize_provider` to accept it.
  In `request_ai_summary`, the local branch POSTs the OpenAI
  **chat-completions** shape to `{ai_local_endpoint_url}/chat/completions`.
  Do not assume the server honors `response_format` json-schema: reuse the
  existing strict validation of the returned text against
  `AI_SUMMARY_JSON_SCHEMA` (already in this module) with **one** retry on
  invalid JSON, then fail the call. The local provider needs no API key —
  use the fixed placeholder `"local"` instead of raising the missing-key
  error. Cloud branches unchanged.
- **The local branch caps evidence to the tier's context budget
  (`evidence_budget_for(provider)`) and verifies the returned
  `usage.prompt_tokens` against an estimate of what was sent, raising a named
  truncation error rather than parsing output generated from a sheared prompt.**
  The check is skipped when the server omits `usage` (some llama.cpp builds do),
  and the error is raised rather than retried: a retry re-sends the evidence
  plus the rejection text, so it would truncate harder. Token usage, when
  reported, is recorded in the job ledger.
- `market_prep/services/ai_service.py`: when a base-URL setting is present,
  pass `base_url=` to the `OpenAI(...)` constructor and fall back to the
  `"local"` placeholder key when none is configured. Cloud path unchanged
  when unset.
- Tests: provider-selection unit tests for both call sites, including
  "settings unset → identical request to today" and "endpoint down → clean
  error, no crash".

### 6.3 New package layout (built up across phases)

```
scripts/ai_jobs/
  __init__.py
  store.py          # ai_store_dir resolution, home-folder-path refusal, layout bootstrap
  window.py         # off-hours window logic (6.1 semantics)
  ledger.py         # append-only JSONL rows → ai_store/logs/ai_job_ledger.jsonl
  runner.py         # named-slot scheduler; slot/state pattern from master_avwap_mini_pc.py
  briefs.py         # Phase 1: AI summary scheduling + per-ticker briefs
  extract.py        # Phase 2: deterministic fact extraction (code only, zero LLM)
  digest.py         # Phase 2: digest writer (LLM narrates around extract.py facts)
  journal_enrich.py # Phase 3: tagging assist, scaffolding, weekly retro
  policy_draft.py   # Phase 4: review_policy_draft.json writer
  cohorts.py        # 2026-08-20: deterministic veto-cohort grading (NO model)
```

Tests land as `tests/test_ai_jobs_*.py` per module. Every job writes a ledger
row (job name, model, duration, token counts if reported, exit status) whether
it succeeds or not; a failed job leaves prior artifacts untouched
(write-temp-verify-rename, the atomic-publish pattern).

### 6.4 Digest schema v1 (SUPERSEDED DRAFT — do not build)

> **2026-08-08: this draft is not the schema to build.** Phase 2 is being
> redesigned around deterministic fact packs (see the Phase 2 status note) and
> that design is pending. Trader decision: do not build or freeze any digest
> schema without a design packet first. Kept below only as the record of what
> was drafted, never signed off, and never written. **The current proposal is
> sec 6.4a below**, itself awaiting trader sign-off.

One JSON object per session day, ≤32KB hard cap, written to
`ai_store/digests/YYYY/YYYY-MM-DD.json`:

```jsonc
{
  "schema_version": 1,
  "session_date": "2026-08-07",
  "generated_at": "2026-08-07T20:15:03-04:00",   // tz-explicit, always
  "sources": [ {"path": "...", "records": 42, "ids": ["..."]} ],
  "facts": { /* deterministic numbers from extract.py ONLY —
               alerts fired/taken/skipped, R stats, scan counts,
               conversion rates. The LLM never writes this object. */ },
  "narrative": { "model": "gemma3:12b",
                 "session_summary": "...",
                 "standouts": ["..."] },
  "tags": ["..."],
  "evidence": [ {"claim": "...", "source": "...", "ref": "..."} ]
}
```

Fields may be **added** in later versions; existing fields are never renamed,
retyped, or removed (append-only ledger, sec 3.2). Malformed or over-cap
output → job fails, no file written.

### 6.4a Phase 2 design packet (PROPOSED — awaiting trader sign-off)

> Supersedes the sec 6.4 draft. **Nothing here is frozen and no code is written.**
> This is the design packet the 2026-08-08 trader decision requires before any
> digest schema may be built. Field names below are illustrative of *shape*, not
> a schema to implement; the open questions at the end must be answered first.

**The problem this solves.** Phase 1's lesson was that everything trustworthy in
the nightly output came from code and everything that needed guarding came from
the model. The 2026-08-10 truncation made the point again: a model handed a
sheared prompt produced confident, schema-valid output about evidence it never
saw. So the extraction layer is the product, and narration is a garnish that
must never be load-bearing.

#### D1 — Two artifacts per session, not one

| Artifact | Written by | Status when the model is unavailable |
|---|---|---|
| `facts/<YYYY>/<YYYY-MM-DD>.json` | code only, zero LLM | **written normally** |
| `narration/<YYYY>/<YYYY-MM-DD>.json` | medium tier, reads only the fact pack | absent |

Splitting them is the whole design. A missing narration file is a normal state,
not a degraded one: the frontier reducer reads facts, and narration is a
convenience for the human. It also means narration can be regenerated later —
after a model upgrade, say — without touching an append-only fact record, and
that a model failure can never block the day's facts from being recorded.

#### D2 — Every number is computed by code and carries its own pointer

No aggregate is ever produced by a model. The proposed shape for a measured
value makes the provenance impossible to omit:

```jsonc
{"value": 1.01, "n": 1940, "source_id": "review.scoreboard",
 "selector": "bucket=favorite_setup&window=60d", "as_of": "2026-08-10T16:05:00-07:00"}
```

`n` is mandatory. The -0.18R vs +1.01R finding that reordered the Away report was
only actionable because both sample sizes were known; a bare average would have
looked like a coin flip either way.

#### D3 — Fields the frontier reader actually needs

Sized to answer "which setups, conditions, and habits are winning" across 30–90
digests without ever reading raw data:

- **Setup performance** — per setup family and per priority bucket: n, win rate,
  mean and median R, and a small distribution (not just the mean, which hides
  the fat tail that makes a setup worth trading).
- **Condition slices** — market environment (`bullish_weak` and friends, already
  stamped on technical-integrity events), time-of-day bucket, and session
  character. A setup that only works in one regime reads as mediocre when
  averaged across all of them.
- **Trader behaviour** — alerts fired vs reviewed vs acted on, decision latency,
  veto-vocabulary counts, and SEEN/TAKEN/SKIPPED conversion. The habits half of
  the mission lives entirely here, and none of it is a market fact.
- **Operations** — scans completed, provider failure/fallback counts, staleness,
  and publish outcomes. Without these, an infrastructure week reads as a bad
  trading week; 2026-08-10 would have looked like "no setups worth pushing"
  rather than "the writer was unconfigured".
- **Coverage and uncertainty** — what was missing, empty, stale, or unfunded that
  day, mirroring the evidence packager's existing honesty. A digest that cannot
  say what it did not see is a digest a reducer will over-trust.

#### D4 — Evidence pointers drill to raw records

Every pointer carries `source_id`, the artifact path, a record selector, and an
`as_of`. A pointer must survive the record moving to the DAS or into an archive
partition, so it names a logical source and key rather than a byte offset.

#### D5 — Sizing, by construction rather than by truncation

- Fact pack: target ≤8 KB, hard cap ~16 KB. 90 of them is well under 1.5 MB —
  a trivial context load for a frontier model, which is the entire point.
- **The narrator reads the fact pack and nothing else.** That bounds its prompt
  to the cap plus a fixed scaffold, which fits `ai_local_evidence_budget_chars`
  (22,000) by construction. Raw sources are never fed to the narrator, so the
  truncation class of failure cannot recur here by design, not by vigilance.
- Over-cap output fails the job and writes nothing rather than truncating.

#### D6 — Append-only, point-in-time, timezone-explicit

Built only from information available that session; every timestamp carries an
explicit offset (plan.md sec 5). A digest is never edited — a correction is a
superseding sibling naming what it supersedes, so the history of what was
believed on the day survives.

#### D7 — No local large model

Narration is medium tier or nothing (sec 2). Any design that needs a 27B-class
local model is out of scope on this hardware.

#### D8 — Rollups are a read, not a second store

Weekly and monthly views are computed on demand from the fact packs. A derived
aggregate store would be a second thing to keep in sync and a second thing to be
wrong.

#### Open questions — the trader must answer these before anything is built

1. **What counts as "winning"?** R at scenario close, MFE/MAE, or both? This is
   the single decision the whole fact pack hangs on, and it is a trading
   judgement, not an engineering one.
2. **Which condition slices are first-class?** Every slice multiplies the fact
   pack; the honest starting set is small and named deliberately.
3. **Do shadow-engine outputs belong in the digest?** They are promotion
   evidence (plan.md sec 7) and mixing them with champion facts risks a reducer
   treating a challenger as live.
4. **Retention:** are narration files disposable and regenerable, or part of the
   permanent record?
5. **Cap:** is 16 KB the right hard cap, given it bounds how much a single day
   can ever say?
6. **Non-sessions:** does a weekend or holiday get an empty fact pack (so gaps
   are visible) or no file (so absence means "no session")?

### 6.4b Ticker-briefs hardening packet (BUILT — armed by trader 2026-08-11; TB-5/TB-6 added 2026-08-12)

> Drafted at trader direction on the evening of 2026-08-10, hours before the
> first repaired 22:00 window, so that a bad night has a ready, reviewed plan
> instead of a 2 a.m. improvisation. **Armed by the trader on 2026-08-11**
> after reading the first overnight run's evidence, and built the same day
> (TB-0 through TB-4). An external frontier-model review (2026-08-10) proposed
> overlapping changes; every claim below was re-verified against the code on
> this branch before inclusion, and the ones that did not survive verification
> are recorded at the end.
>
> **What the first night actually proved (2026-08-10/11).** `ticker_briefs`
> completed all 95 symbols in 5,962 s — **~63 s/call**, not the ~4.75 min/call
> the packet was drafted against. Defect 1 below is therefore **obsolete as
> written**: the batch fits the window comfortably and there was no overrun.
> It is kept, struck through, because the correction is the evidence. The real
> finding was **content vacuity**, recorded as TB-0.

**Verified defects this packet repairs.** Confirmed by direct inspection of
`scripts/ai_jobs/briefs.py` and `scripts/ai_jobs/runner.py`. The first two are
the open defects already named in `CURRENT_CHECKPOINT.md`; the rest are their
sharper consequences:

0. **TB-0 — every brief was content-free** (found 2026-08-11, added at arming
   time; the highest-value defect in the packet). `run_ticker_briefs` built one
   base evidence package *already budgeted to the local ceiling*
   (`evidence_budget_for("local")` = 22,000 chars) and then projected each
   symbol out of that starved base. The budget pass had marked the
   per-symbol-rich sources unfunded at 0 chars — `setups.current_tracker`
   95,806 chars, `setups.current_tiers` 77,124, `setups.bounce_learning`
   17,995, `market.industry_intraday_rs` 17,833 — and sheared the funded tables
   to about one row (`showing most recent 1 of 192 rows`, `1 of 200`). By the
   time the projection ran, the symbol was no longer in the package. The MRVL
   brief's coverage says **"1 of 19 requested source(s) usable"**, and the one
   usable source was `watchlists.membership`. All 95 briefs were structurally
   like that: 5,962 s of inference describing which lists a ticker is on.
1. ~~**`ticker_briefs` cannot finish as scoped.**~~ *(obsolete — see the note
   above; measured at ~63 s/call, ~99 min for 95 symbols.)* One model call per
   unique Focus/watchlist symbol — 95 on 2026-08-10, deduplicated across all
   six lists — at an assumed ~4.75 min/call would be ~7.5 hours against an
   8-hour window, in a slot reserving 120 minutes.
2. **A failing job has no attempt cap.** Only `ok` is canonical, so a
   deterministic failure retries on every 30-minute firing to the end of the
   window (11 consecutive failures, ~111 minutes of inference producing
   nothing, on 2026-08-09/10).
3. **No per-ticker error isolation.** The symbol loop in `run_ticker_briefs`
   calls `request_ai_summary` exactly once per symbol with no try/except and
   no retry — the daily summary's two-attempt fed-back-error loop has no
   counterpart here. One validation failure, timeout, or mid-batch window
   closure raises out of the entire job.
4. **All-or-nothing morning file.** `ai_morning_brief.txt` publishes only
   after every symbol succeeds. At 99% per-call reliability the chance all 95
   complete is ~39%; a night that briefed 94 names publishes nothing.
5. **Retries regenerate everything and duplicate artifacts.** Completion is
   ledgered per job, not per symbol, so the next firing restarts at symbol 1.
   Export filenames are wall-clock stamped (`ai_summary.export_ai_summary`),
   so each partial attempt leaves a second full four-file artifact set for
   every symbol it re-completed.
6. **Membership-only symbols still get a full model call.** The projected
   package always contains the `watchlists.membership` source, so a symbol
   with no other evidence spends ~5 medium-tier minutes paraphrasing "it is
   on swing_longs" — the class of output least likely to say anything.

#### Work items, ordered by expected value

**TB-0 — Project first, budget second.** *(built)* For the ticker-briefs path
only, build the base package with the **cloud** ceiling
(`MAX_TOTAL_EVIDENCE_CHARS`) so symbol rows survive into projection —
`_extract_ticker_content` already bounds each projected source at
`MAX_TICKER_SOURCE_CHARS` = 16,000 — then apply the **local** budget
(`evidence_budget_for("local", tier="medium")`) to each projected per-symbol
package before the model call. A per-symbol package still over budget is
rationed with the packager's own unfunded/truncation vocabulary
(`ai_summary.ration_projected_sources`), so the truncation tripwire keeps
holding for every local call. `run_daily_summary` is untouched: it still
budgets its one package to the local ceiling, and cloud request payloads stay
byte-identical. Projections copy their sources, notices included, so one
symbol's truncation banner can never land on the base the next symbol reads.

*Residual, stated honestly:* an 80,000-char base is a ceiling too. A single
95,806-char tracker inside a four-scope package still gets a weighted share
(~24,000 chars before surplus reallocation) and keeps its **most recent** rows,
so a symbol whose only row is an old one can still be sheared out before
projection. That is a large improvement over being unfunded at 0 chars, not a
guarantee. Raising the base ceiling further is a trader decision and belongs
with the fact-pack design (sec 6.4a), not here.

**TB-1 — Per-ticker failure isolation and an honest partial morning file.**
Wrap each symbol's inference/export in per-symbol error capture; give each
call the same single fed-back-error retry the daily summary already has. The
morning file publishes whatever completed, with the outcome stated before any
brief: `Briefed N of M. Failed: SYM (reason), …` in the header block, so a
partial file can never be mistaken for a complete one. Focus symbols already
lead the ordering (`default_watchlist_paths` is Focus-first), so a partial
night preferentially covers Focus — this subsumes the "separate Focus
publication unit" idea. Job status is `ok` only when all symbols resolved;
otherwise `degraded`, which the runner already retries. A mid-batch off-hours
window closure stops inference (the hard rule is untouched) but publishes the
honest partial file instead of losing the night; the market-session block
remains an unconditional stop. The existing refusal when watchlists are
unreadable-and-empty stays exactly as is.

**TB-2 — Skip membership-only symbols deterministically.** If a symbol's
projected package contains no usable source beyond `watchlists.membership`,
do not call the model: emit a deterministic one-line morning-file entry
("no session evidence beyond membership in swing_longs") and no ai_store
artifact set. Counts as resolved for TB-1's N-of-M. Expected effect: tonight's
95 calls shrink to roughly the 10–30 symbols that actually appear in reports,
trackers, or the journal — the single largest runtime and reliability lever
in the packet.

**TB-3 — Resumable completion keyed by (session_date, symbol, evidence_hash).**
Record per-symbol completions in a per-session manifest under
`ai_store/briefs/<year>/<session>/`; on re-fire, skip symbols already
completed for the same `evidence_hash` and regenerate only when the hash
changed. Ends both the restart-at-symbol-1 waste and the duplicate artifact
sets. The morning file is re-rendered from the manifest each time, so a
retry that clears the failures upgrades `degraded` to `ok` naturally.

**TB-4 — Per-session attempt cap.** Cap `ticker_briefs` (and any future long
slot) at 2–3 attempts per session; on reaching the cap, record a terminal
marker the runner respects so remaining firings skip in ~a second, the way
no-session firings already do. Transient self-heal (NAS asleep, endpoint
down) survives; the all-night grind does not. An identical-error early stop
(same exception text twice → stop for the night) is an optional refinement
inside this item, not a separate mechanism.

**TB-5 — A roster line is not evidence about the symbol.** *(built 2026-08-12,
after the packet's first live night)* TB-0 moved the budgeting, and the usable
source count per brief rose from 1 to a median of 3. The count was misleading.
`_extract_ticker_content` projects a text source by keeping every **line**
containing the symbol, and the evidence files are human-readable reports full of
copy-paste ticker blobs, so most of what it matched was rosters: MDB "appears in"
`daily.master_events` because MDB is one of ~300 names in a `LONG: A, AAPL, …`
dump. Measured across all 166 briefed symbols of 2026-08-11:

| | chars | share |
|---|---:|---:|
| Roster / name-dump lines | 307,630 | **96.2%** |
| Everything else | 12,057 | 3.8% |

Median symbol-specific content was **42 characters**, and much of even that was
membership restated. `setups.current_tracker` — the source TB-0 was built to
rescue — contributed **zero** symbol-specific content and 10,368 chars of pure
roster. Only **18 of 166** symbols had a real scan line (`rs=`, `1d`, `5d`) and
**6** a tier observation. The output followed the input: MDB's brief attributed
its 1d/5d figures to `daily.master_events` (a roster line; the numbers are in
`daily.market_prep`), called `setups.tier_performance` truncated, and never
reported the one genuinely useful number it had been handed,
`MDB 2026-08-04->2026-08-11 (+15.56%)`.

The rule is about **residue, not ticker count**: strip the ticker tokens and list
punctuation and see whether anything is left. A tier row carries eight tickers and
is pure signal, so a count threshold would have discarded exactly the rows worth
keeping. A bare-symbol line is dropped for the same reason TB-2 exists — Auto
Pilot's `longs` array saying `"MDB"` is watchlist membership wearing a second hat,
and counting it as evidence kept symbols out of the membership-only skip while
telling the model nothing. Measured effect on the same data: **166 model calls →
49**, projected payload 319,687 → 26,223 chars.

*Residual, stated honestly:* `setups.current_tracker` arrives as a single-line
JSON string, so line-based projection remains all-or-nothing for it — two symbols
still receive ~5,184 chars of tracker covering mostly other names. A real fix is
structured extraction (parse, select the symbol's setups), not a grep, and belongs
with the sec 6.4a fact-pack design rather than here.

**TB-3 repair — resume on the evidence, not on when it was read.** *(built
2026-08-12)* The defect recorded at build time fired on the first live night. The
manifest now carries a `resume_key` hashing only symbol, session, memberships, and
source ids with their content; `evidence_hash` keeps its whole-package meaning for
artifact identity. Manifest schema `v1` → `v2`, and a row without a `resume_key` is
regenerated rather than reused — an older manifest costs a regeneration, never a
wrong skip.

**TB-6 — Publication survives a hard kill.** *(built 2026-08-12)* TB-1 made the
morning file honest about partial nights, but it was still written once, after the
loop. On 2026-08-11 the desk entered Modern Standby mid-batch and the process died
at symbol 101 of 182: 126 briefs existed in the AI store and the home folder still
held the previous session's file. The file is now re-rendered and atomically
republished after every resolved symbol, carrying an explicit "Run in progress at
the time of writing" note that the final publish drops. A publish fault is logged
and never costs the batch. **The market-session block still suppresses publication
outright** — it is an unconditional stop for the whole job, publication included,
and the last verified file stands.

#### Explicitly deferred — recorded so they are not re-proposed as new ideas

- **Changed-only swing narration** (brief a swing name only when its state
  changed): needs durable prior-session state and a change definition; real
  machinery for a layer still in its first trial. Revisit after five clean
  sessions with TB-2's filtering evidence in hand.
- **Compact ticker-specific output schema** (~400–700 output tokens): real but
  second-order once TB-2 removes the evidence-free calls. Fold into the
  Phase 2 fact-pack design (sec 6.4a) rather than patching the generic schema.
- **Parallel local inference**: rejected outright — calls would compete for
  the same UMA/Vulkan resources; sequential stays.

#### Frontier-review claims that did not survive verification

- "The built-in retry improves batch reliability" — false for `ticker_briefs`;
  only the daily summary retries. Corrected by TB-1.
- Its ~1 min/ticker estimate — rejected at drafting time against a desk-observed
  ~4.75 min/call. **The frontier review was right and this rejection was wrong**:
  the first repaired night measured **~63 s/call**, which is its estimate almost
  exactly. The ~4.75 min figure came from the pre-repair model. Defect 1 is
  therefore not a window overrun, and TB-0 — which neither review named — was
  the defect that mattered.

#### Gate interaction and scope

Arming this packet changed `ticker_briefs`' failure policy, and the Phase 1
exit gate's own reset conditions say changing the observed thing restarts the
observation. **Adopted at arming (2026-08-11):** `ai_summary` and
`ticker_briefs` are judged on **separate five-session clocks** — the daily
summary's code path is untouched by TB-0..TB-4, so **its clock continues**;
the **ticker-briefs clock restarts at zero**.

Scope as built: `scripts/ai_jobs/briefs.py`, `runner.py`, `ledger.py`, one
additive helper in `scripts/ai_summary.py` (`ration_projected_sources`, called
only from the ticker path), plus tests. No detector, scoring, or alert file is
touched, and every output remains advisory-only with zero influence on
scanners, scores, watchlists, alerts, or bot state.

Tests landed with the build: TB-0's project-then-budget proof (the row a
budget-first base loses, and that the same row survives the new order), every
per-symbol package staying inside the local budget, the daily-summary
package unchanged, projections not mutating their base, partial-publish header
rendering, the unreadable-watchlist refusal unchanged, membership-only skip,
resume skipping same-hash completions and regenerating changed-hash ones,
attempt-cap and identical-error terminal behavior, cheap skips never spending
an attempt, `--force` overriding the marker, and window-closure-mid-batch
publishing the partial file while the market session still stops outright.

**Live proof still owed:** the next 22:00 window on the desk. What to check in
the morning — briefs whose coverage counts more than one usable source and
whose statements cite real evidence; a morning-file header reading
`Briefed N of M`; at most three `ticker_briefs` ledger rows for the session,
with a `terminal` row if it stopped; and no duplicate artifact sets under
`ai_store/briefs/<year>/<session>/tickers/<symbol>/`.

**Implementation notes worth keeping.** The per-session completion manifest is
`ai_store/briefs/<year>/<session>/ticker_briefs_manifest.jsonl`, append-only,
newest row per symbol wins; an unreadable manifest regenerates the night
rather than refusing it. The attempt cap is a `JobSlot.max_attempts` the runner
enforces, and its terminal marker is an ordinary `skipped` row carrying
`terminal: true` — deliberately **not** a new job status, because
`RECOGNISED_JOB_STATUSES` governs what a *job* may report and this row is
written by the runner. Only `failed` and `degraded_no_narrative` rows spend an
attempt: a `skipped` refusal (unmounted Drive) costs about a second and must
keep self-healing.

### 6.4c Nightly journal pull (QUEUED — trader-approved 2026-08-11, do not build yet)

> The trader approved queuing this on 2026-08-11 and explicitly deferred the
> build ("we wont do it yet"). Do not implement it before the ticker-briefs
> hardening packet (sec 6.4b) has its live proof and the trader says go. It is
> recorded now so the design survives the wait.

**What it is.** A third Phase 1 runner slot, `journal_import`, that pulls
broker executions unattended each night so the nightly `ai_summary` narrates a
journal that already contains the session's trades — today that freshness
depends on the trader remembering the GUI Broker Sync button. The AI side
needs zero changes: `journal_import_health` (lag, newest execution) is already
read and published every night.

**Why it is cheap.** The import path is already headless and already safe to
repeat: `scripts/journal_runner.py` has no Qt dependency (the GUI button just
wraps it in a thread), executions dedupe on `execution_uid` by its own
docstring, and every pull records an import-run row win or lose. The slot
reuses `run_journal_import_for_date` / `run_journal_backfill` as-is and
inherits the runner's ledger idempotency, 30-minute self-heal, and the sec
6.4b attempt cap. It is a seconds-scale network call, not inference: reserve
~5 minutes, `max_attempts=3`.

**Design decisions to honour when building:**

1. **Slot order: `journal_import` runs FIRST, before `ai_summary`.** The
   runner's slate comment says later phases append and never reorder; placing
   this slot ahead of the summary is a deliberate, documented exception — the
   entire point is pull-before-summarize.
2. **IBKR path is Flex, not the socket.** The socket API returns only the
   current session's fills, which after TWS's daily reset at ~22:00 is likely
   nothing. The Flex Query web service (`run_journal_backfill` already
   supports it) returns the complete statement at any hour. Nightly shape:
   Questrade for recent days + IBKR Flex when the `journal_ibkr_flex_token` /
   `journal_ibkr_flex_query_id` settings are configured; Questrade only
   otherwise. The socket importer stays a desk-hours/manual path.
3. **Questrade token rotation race, stated not solved.** The refresh flow
   saves a new single-use refresh token on every pull. A nightly pull racing a
   manual GUI sync could invalidate the token; at 22:00+ this is nearly
   theoretical. Worst case is a FAILED import row and a one-time re-auth —
   never silent corruption. Do not add locking for it.
4. **One-writer statement.** The journal SQLite lives in the home folder and
   the GUI also writes it. The nightly slot owns *unattended* imports; the GUI
   button remains the manual path. A rare write collision surfaces as a
   FAILED run and self-heals on the next firing — acceptable, documented,
   no new locking.
5. **A zero-execution session is a normal `ok`,** not degraded: "imported 0,
   rebuilt N trades" is a true statement about a day the trader did not trade.

**Tests owed with the build:** slot registration and ordering ahead of
`ai_summary`; per-session idempotency through the ledger; Flex-vs-socket
selection by settings; a FAILED import recording honestly and retrying on the
next firing without touching prior journal rows; a zero-execution night going
`ok`. No detector, scoring, or alert file is in scope; the journal store's
trader-entered fields are never touched by this slot (it writes executions,
accounts, and grouped-trade rebuilds only, exactly as the manual path does).

### 6.5 Phase exit gates (verify commands)

Every phase: `.venv\Scripts\python.exe -m pytest tests/ -q` fully green
(pytest's own exit code) and `scripts/smoke_check.py` 7/7 before merge.
Additionally:

- **Phase 0:** with settings set, `ai_summary` returns a schema-valid summary
  from the local endpoint; with settings unset, request payloads to cloud
  providers are byte-identical to pre-change (test-asserted). Benchmark table
  (tok/s + RAM per tier on the 8845HS) recorded in this doc.
- **Phase 1:** one week of unattended morning summaries + briefs, zero manual
  action, every run in the ledger.
- **Phase 2:** 10 consecutive session-day digests; trader spot-audit of ≥3
  finds zero fabricated facts (every number traceable to `extract.py`).
- **Phase 3:** LLM tags land only in advisory fields via the `JournalStore`
  API; trader-entered data provably untouched (test: enrichment run leaves
  all non-advisory columns bit-identical).
- **Phase 4:** two weeks of `review_policy_draft.json` side-by-side vs the
  cloud model; trader sign-off recorded here before the live file is ever
  written locally. Rank/annotate only; no suppression field (test-asserted
  against the schema).
- **Phase 5:** synthesis report cites only digest/retro evidence pointers;
  no behavioral recommendation ships anywhere except the report text.

## 7. Confirmation register (open items)

1. **Machine roles — RESOLVED 2026-08-08:** the 8845HS is the main desk; all
   AI roles live on it (see sec 2). The former desk (i5-8600K + 3080 Ti) is
   powered down most days — power draw, office heat, and workstation/gaming
   tax separation — and serves as the discord/chat box with, at most, an
   ad-hoc alternative-scanner role. It must never be assigned an always-on
   job, a writer lease, or the inference-host role. The separate
   `master_avwap_mini_pc.py` scanner-machine role is **retired**
   (trader decision 2026-08-08): the 8845HS is both the mini-PC and the main
   PC, so there is no second scan host and no cross-machine IB budget
   question. The script stays in-repo as the slot/state scheduling template
   this plan reuses (sec 3.4); whether to delete it outright is a separate
   cleanup decision.
2. **File server path class — RESOLVED 2026-08-10** (sec 6.1:
   `ai_store_dir` accepts any path outside the home folder, local disk included).
   `research_warehouse/config.py` accepts a UNC path: `research_store_dir` is now
   `\MINI-PC\Trading Bot Data
esearch_lake`, and `ensure_lake_layout` created the
   full sec-8.2 skeleton over SMB. The trader confirmed the DAS is the durable
   storage tier (decision 0015), so lake and AI store both live there.
3. **Model picks — RESOLVED BY MEASUREMENT 2026-08-08, PARTLY OVERTURNED
   2026-08-10** (sec 6.1; see the amendment note at the end of this item):
   `gemma3:4b` (small) and `gemma3:12b` (medium) both run 100% on the 780M.
   Stock `gemma3:27b` does not fit the 17.4 GiB Vulkan heap, so the large tier
   is `hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M` — verified loading
   and producing schema-valid output. All three are settings, and Phase 0
   finding 2 records the revisit triggers.

   **Amended 2026-08-10 by measurement, twice:**
   (a) the medium tier is now the derived `gemma3:12b-tbv3ctx` (`FROM gemma3:12b`,
   `PARAMETER num_ctx 12288`). The stock tag inherits the server default, which
   measured **2,048 prompt tokens** — so every `ticker_briefs` run silently had its
   evidence truncated and returned unterminated JSON, failing six nights running. The
   derived model measures **6,147**. Per-model `num_ctx` is used rather than the global
   `OLLAMA_CONTEXT_LENGTH` because one global value cannot serve both tiers here: 16384
   fails to allocate outright, and a value large enough for briefs starves the 27B.
   (b) the large tier **currently fails to load at all** while the desk is running —
   `alloc_tensor_range: failed to allocate Vulkan0 buffer` — even at default context.
   Phase 0's verification was done under different memory conditions. No large-tier job
   is scheduled yet, so nothing regressed; but Phase 2 has no working large model on
   this hardware until it is re-sized or run with the desk closed.
4. **Off-hours window edges — SET BY THE TRADER 2026-08-08**: **01:00–09:00
   ET**, which is the 22:00–06:00 the trader asked for on this Pacific desk
   (Pacific is ET−3 in both DST and standard time, so the mapping is stable
   year-round). Weekends open, holidays treated as weekdays. The defaults in
   sec 6.1 remain 18:30–08:00 for an unconfigured machine.

   Noted and accepted: 09:00 ET is 30 minutes before the opening bell. The
   trader was shown this and reaffirmed the choice. It is bounded rather than
   argued: the market-session block refuses the session itself regardless of
   the window, `launch_allowed(reserve_minutes=...)` refuses to *start* a job
   that cannot finish (the summary slot reserves 20 minutes), and the worst
   measured single model call is ~4 minutes, so the residual exposure is one
   in-flight call finishing gracefully well before 09:30.

4b. **AI store location — SET BY THE TRADER 2026-08-08**:
   `\\MINI-PC\Trading Bot Data\ai_store` on the NAS. Verified reachable and
   writable, layout bootstrapped, atomic publish (temp → `os.replace`) proven
   on the share. Measured: **19.8 s first write** while the NAS spins up, then
   ~40 ms per fsync'd append and 55 MiB/s write / 78 MiB/s read. Placed in a
   dedicated `ai_store` subfolder rather than the share root, which already
   holds `data/`, `logs/` and `output/` — the AI store's own `logs/` would
   otherwise have collided with an existing directory. The live operational
   home folder is `C:\TradingBotData`, so the sec 3.3 "never inside the synced
   home folder" rule is satisfied.
5. **Digest schema v1 — DRAFTED** (sec 6.4); trader sign-off on the field
   list is still required before the first ledger write (append-only from
   then on; later fields extend, never mutate).

---

## Amended 2026-08-09 — repair packet 2 (Sol 5.6 verification review)

Verifying the 2026-08-08 repairs found that several of them were warnings
rather than controls, and one had never reached production at all. What
changed, in this layer:

- **Session identity** now comes from a real NYSE calendar
  (`scripts/market_calendar.py`) and **fails closed**. A run resolves the most
  recent session whose close is at or before run time; a weekend or holiday
  firing works on the last completed session, or records one no-session skip
  row if that session is already covered. Nothing is ever keyed to a date the
  exchange did not open.
- **`manual_test`** is a distinct ledger status for `--force` and operator
  runs. It publishes real artifacts and never satisfies the
  canonical-completion check, so the scheduled run still happens.
- **`correction`** is a distinct ledger status for retracting a coverage claim
  by appending, never by rewriting. The ledger stays append-only.
- **The journal is session-scoped and import-aware.** Filtered to the session
  by SQL; a stalled import makes it stale and hides its old rows; import
  health (last import, newest execution, lag days, session row count) is
  reported as a `[system]` data-quality row.
- **`data_quality` is machine-owned** — out of the model's schema, out of the
  prompt, rejected by the validator if returned. The model keeps `risk_notes`.
- **Stale sources leave the package** rather than carrying a warning into it.
- **Unknown job statuses fail closed** to `failed`.
- **The pre-open reserve is 15 minutes**, inside the session block, so
  `--force` cannot spend it.
- **Reads are bounded** and the raw setup tracker is packaged as a
  most-recent extract rather than a head slice of March watchlists; within
  `setup_trackers`, analytic sub-sources are funded before the raw tracker.

**Closed 2026-08-10:** the 80,000-character evidence budget is no longer used
for local calls. `ai_local_evidence_budget_chars` (default 22,000) caps the
local branch to its context window, and a truncation tripwire fails loudly when
the server reports having seen materially less prompt than was sent. 80,000
remains the cloud ceiling, so metered models are not penalised by a local limit.

Still open: the tracker file itself is 762 MB.


---

## 7. Addendum, 2026-08-20 — a deterministic slot, and an opt-in scope

Two changes to this layer, both trader-authorized, both outside the phase
ladder above because neither is an inference job.

### 7.1 A fourth slot: `veto_cohort_grading`

`ui.annotations.veto_cohort.update_veto_cohort_outcomes` shipped with the
Chart Review packet and had **zero callers** from that day until now. Veto
picks accumulated on every capture-rail commit and nothing ever graded them,
so "are my vetoes any good?" stayed computable-but-unanswered.
`ai_jobs/cohorts.py` is the caller.

**It is not an AI job.** No model is consulted (a test asserts the local
provider is never even reached), nothing is transmitted, and the output is two
CSVs. It lives here because this is where the desk's overnight slate runs.

Registered by **appending** to `default_slots()`, per that function's own rule
— *"later phases append; they never reorder these."* A slot rather than a step
inside `journal_import`, because the slot is the unit this runner already
gives every job: its own ledger row, retry budget, reserve check and failure
isolation. Folding it in would make a grading failure read as a journal
failure. Placed **last** at a 5-minute reserve because it costs seconds,
nothing downstream reads it, and the briefs must not lose window time to it.

Contract, tested:

- **Idempotent in the sense that matters.** A re-run the same night moves
  exactly one column, `updated_at`, and nothing measured. Byte-identical is
  deliberately not claimed — a provenance stamp is supposed to move. A fully
  matured pick is never recomputed at all.
- **A failure leaves both CSVs byte-identical**, per this section's own
  atomic-publish rule.
- **Sideless picks are counted and named, never graded.**
  `human_focus_tracking._side_label` reads a blank side as LONG, so grading one
  would manufacture a directional claim the trader never made.
- The forward return is **close-to-close only** — it does not read volume or
  AVWAP bands, so the known IBKR/Yahoo volume-unit defect does not reach these
  numbers.

First desk run: 45 picks → 44 graded rows, 0 sideless, 0 cohorts (nothing had
matured on day one).

### 7.2 `trader_judgement`: a scope that is registered but not nightly

The capture rail's stream (`trader_annotations.jsonl` — vetoes, setup claims,
notes) had **no overnight reader at all**. It now has a scope in
`ai_summary.py`, with sources in funding order: the veto performance rollup,
then the outcomes, then the raw annotation log **last** (§6.4b's rule — the
raw tracker leading its own scope is what starved every analysis derived from
it).

**Deliberately absent from `DEFAULT_SCOPES` and `TICKER_BRIEF_SCOPES.`**
`pick_feedback` is the precedent for a registered-but-not-nightly scope. The
reason here is evidence density, not caution: veto cohorts need forward
returns before they mean anything, so an unattended nightly read would narrate
"too early" over a stream still filling.

Exercised on demand: `run_ai_jobs.py --scopes trader_judgement`. The override
is constructed per call, so the unattended path passes nothing and gets
`DEFAULT_SCOPES` — an opt-in scope cannot leak into the slate by being set
once. Unknown names are rejected at the CLI against the registry.

Two **machine-written caveats** travel with the scope in the package as data,
in the same sense `coverage` is — exact, code-owned, never inferred. Both are
properties of the capture UI rather than of the trader's judgement, and a
reader without them draws a confident wrong conclusion from a correct file:
the like+claim control currently offers only the "Main swing" group, and the
"Veto D1 — but M5 today" verb writes an ordinary veto row so some vetoed names
were traded the same day.

### 7.3 What is NOT built

A weekly synthesis pass over the graded cohort. The **cadence is decided**
(weekly, on the weekend surface — recorded against R8 in `plan.md`) and it is
**gated on two weeks of graded rows**. It is not authorized. Nor is any
frontier call, any nightly LLM read of the raw annotation stream, or any path
by which these files reach a detector, a score, an alert, a watchlist, Focus,
the review queue or `review_policy.json`.

