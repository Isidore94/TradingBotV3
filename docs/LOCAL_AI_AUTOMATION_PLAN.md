# Local AI & Automation Plan

Status: ACCEPTED into plan.md sec 12 as item 13b (trader-directed,
2026-08-08). **Phase 0 COMPLETE on branch `local-ai-phase-0` (2026-08-08):
code landed, Ollama installed and benchmarked on the main desk, all three
tiers chosen and verified, and the exit gate verified end to end. Phases 1+
not started.** Subordinate to `plan.md` — this
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
| Medium (12-14B Q4) | Gemma 3 12B / Qwen3 14B | nightly digests, journal summaries, briefs |
| Large (27B+ Q4, ~18GB via UMA) | Gemma 3 27B class | review-policy drafting, weekly retros — the reasoning-heavy overnight jobs |
| Frontier (cloud, metered) | Fable 5 / best available | periodic synthesis over digests only |

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

- **Not in the Drive home folder** (sync churn on nightly bulk writes) and
  **not inside the DAS lake tables** — the lake and the AI store are separate
  storage classes with separate writer components. Both components now live
  on the same main desk, which is fine: ownership is per-component, not
  per-machine, and keeping the trees separate means an AI-job bug can never
  corrupt lake data.
- Small human-facing outputs only (morning brief, weekly retro) additionally
  publish to the Drive home folder via the existing atomic-publish pattern:
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
  `ai_store/briefs/` + small morning file to Drive.
- Exit gate: a week of mornings where the summary and briefs are waiting
  before pre-market prep with zero manual action.

### Phase 2 — Daily Digest Ledger (foundation)

- Deterministic extraction layer (code, no LLM): pull the day's facts from
  the sources in 3.2 into a typed intermediate.
- Digest writer: medium model narrates/tags around the extracted facts;
  schema-versioned JSON with evidence pointers; append-only under
  `ai_store/digests/`.
- Exit gate: 10 consecutive session days of digests; trader spot-audits ≥3
  against raw evidence and finds no fabricated facts (numbers all traceable
  to the deterministic layer).

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
  `review_policy.json` rank/annotate output) moves to the local large model,
  nightly.
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
| One owner per mutable export; failed publish never destroys last verified | Main desk solely owns `ai_store`; Drive copies use atomic publish; `autopilot_today.txt` and lake writers unchanged |
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
| `ai_local_model_medium` | `gemma3:12b` | digests, briefs, summaries |
| `ai_local_model_large` | `gemma3:27b` | policy drafts, retros |
| `ai_store_dir` | unset = AI store + all jobs disabled | file-server or local path; **refuse any path inside the Drive home folder**, mirroring `research_warehouse/config.py`'s refusal. A local-disk path is fine while the file server pends — implementation never blocks on server setup |
| `ai_offhours_start` / `ai_offhours_end` | `"18:30"` / `"08:00"` | ET wall-clock (`zoneinfo`, `America/New_York`) job-launch window. Weekends: all day allowed. Holidays treated as normal weekdays (conservative — the window still applies). No job **launches** outside the window; a job that crosses the end finishes its current model call and stops gracefully |

Model tags are starting picks; the Phase 0 benchmark may swap them by editing
these settings — never by hardcoding.

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
  store.py          # ai_store_dir resolution, Drive-path refusal, layout bootstrap
  window.py         # off-hours window logic (6.1 semantics)
  ledger.py         # append-only JSONL rows → ai_store/logs/ai_job_ledger.jsonl
  runner.py         # named-slot scheduler; slot/state pattern from master_avwap_mini_pc.py
  briefs.py         # Phase 1: AI summary scheduling + per-ticker briefs
  extract.py        # Phase 2: deterministic fact extraction (code only, zero LLM)
  digest.py         # Phase 2: digest writer (LLM narrates around extract.py facts)
  journal_enrich.py # Phase 3: tagging assist, scaffolding, weekly retro
  policy_draft.py   # Phase 4: review_policy_draft.json writer
```

Tests land as `tests/test_ai_jobs_*.py` per module. Every job writes a ledger
row (job name, model, duration, token counts if reported, exit status) whether
it succeeds or not; a failed job leaves prior artifacts untouched
(write-temp-verify-rename, the atomic-publish pattern).

### 6.4 Digest schema v1 (draft — trader sign-off required before the first ledger write)

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
2. **File server path class — DOES NOT BLOCK implementation** (sec 6.1:
   `ai_store_dir` accepts any non-Drive path, local disk included). Still
   open as an ops question: confirm `research_warehouse/config.py` accepts a
   UNC/mapped file-server path for `research_store_dir`, and whether the
   trader wants the DAS lake moved to the file server.
3. **Model picks — RESOLVED BY MEASUREMENT 2026-08-08** (sec 6.1):
   `gemma3:4b` (small) and `gemma3:12b` (medium) both run 100% on the 780M.
   Stock `gemma3:27b` does not fit the 17.4 GiB Vulkan heap, so the large tier
   is `hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M` — verified loading
   and producing schema-valid output. All three are settings, and Phase 0
   finding 2 records the revisit triggers.
4. **Off-hours window edges — RESOLVED BY DEFAULT** (sec 6.1): 18:30–08:00
   ET, weekends open, holidays treated as weekdays; trader-adjustable
   settings, not code.
5. **Digest schema v1 — DRAFTED** (sec 6.4); trader sign-off on the field
   list is still required before the first ledger write (append-only from
   then on; later fields extend, never mutate).
