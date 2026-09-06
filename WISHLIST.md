# TradingBotV3 integration wishlist

Last reconciled: **2026-08-15**

This file is the trader-visible parking lot for ideas that may be useful but are not
authorized build work. The authoritative implementation order is `plan.md`.

## Rules

- An AI may suggest, clarify, compare, or estimate a wishlist item.
- An AI must not implement an item from this file.
- Only the trader may promote an item into `plan.md`.
- Promotion requires a defined user outcome, prerequisites, scope, invariants, tests,
  and an insertion point in the roadmap.
- Moving an item to the roadmap changes its status here to `ROADMAP`; it is not
  deleted, so the decision remains visible.
- Retired ideas stay recorded to prevent accidental resurrection.

Statuses:

| Status | Meaning |
|---|---|
| `ROADMAP` | Accepted and ordered in `plan.md`; follow the roadmap, not this file |
| `CANDIDATE` | Worth discussing; not authorized |
| `TRIGGERED_LATER` | Consider only if the named condition occurs |
| `RETIRED` | Deliberately abandoned |
| `PERMANENT_NO` | Conflicts with the product boundary or a hard invariant |

## Accepted ideas already on the roadmap

These are shown here only as a readable product wishlist. Their actual order and
requirements live in `plan.md`.

| Idea | Status | Roadmap location |
|---|---|---|
| Live intraday market-commentary journal with nightly advisory summary | `ROADMAP` | P3.5 |
| Research warehouse pilot and trustworthy setup/style readouts | `ROADMAP` | P3.2, P6.1 |
| Deterministic local-AI fact packs, journal enrichment, policy drafts, and frontier synthesis | `ROADMAP` | P3.3, P6.3 |
| Canonical Opportunity snapshot and identity graph | `ROADMAP` | P4.1–P4.2 |
| Greatness readiness stack and dedicated monitoring lane | `ROADMAP` | P2.6, P4.3 |
| Advisory Command Center and Focus Workbench | `ROADMAP` | P4.4 |
| Typed, deduplicated alert ladder with independent delivery canary | `ROADMAP` | P5.1–P5.2 |
| One snapshot across Desk, Focus, Alerts, Away, journal, and AI | `ROADMAP` | P5.3 |
| Full journal lifecycle and Learning Center | `ROADMAP` | P5.4–P5.5 |
| Screenshot linkage for reconstructable journal/review events | `ROADMAP` | P5.4 |
| Controlled Suggested-universe intake and discovery-source measurement | `ROADMAP` | P5.5 |
| Explicit warehouse bounce-event linkage and tracker-to-detection adapter | `ROADMAP` | P3.2 |
| One-at-a-time setup-family research and promotion | `ROADMAP` | P6.2 |
| Complete Market Prep migration into Qt | `ROADMAP` | P6.4 |
| Clean-machine recovery, installer, icon, and release polish | `ROADMAP` | P7.1–P7.2 |
| Read-only additional broker/data adapters after provider consolidation | `ROADMAP` | P7.3 |
| Trader refinement packets R1–R6 (the 2026-08-14 desk requests) | `ROADMAP` | Phase 0.5 |
| Deep-link a symbol into an external charting tool | `BUILT` | Promoted trader-directed 2026-08-18; built the same day |
| Tax-grade journal reliability (both brokers) + TradesViz/TraderSync-style Journal tab | `BUILT` | Phase 0.5 R7 — built 2026-08-15, deterministic gates green; the spec's six live gates are owed and start after Monday's validation day |
| Weekend Prep guided routine with H1/D1/Monthly strength discovery and weekly auto-tag review | `BUILT` | Phase 0.5 R8 — built 2026-08-15, deterministic gates green; the one-real-weekend live proof is owed |

## Candidate user-experience integrations

**Triage of 2026-08-18 (trader integration redirect).** Every item in this and
the next section was assessed for whether it is buildable from its description
plus the codebase. One was — the TradingView deep link, now `BUILT`. Each of the
rest turned out to need exactly one trader judgment whose plausible answers lead
to *different code*; those questions are written down, one per item, in
the **Open questions** section at the end of this file rather than guessed at. Nothing else here was
implemented, and nothing here was promoted into `plan.md`.

| Idea | Status | Value | Prerequisite or decision needed |
|---|---|---|---|
| Voice dictation for the live commentary journal | `CANDIDATE` | Faster capture while watching charts | Choose local vs cloud speech, privacy, correction workflow, and storage format after P3.5 |
| Deep-link a symbol/timeframe into TradingView or TC2000 | `BUILT` | Faster transition to external deep TA | **Promoted and built trader-directed 2026-08-18.** TradingView link built (`scripts/external_chart_links.py`, "Open in TradingView" on the arm bar, URL template is a machine-local setting); TC2000 deliberately not wired because it answers no documented URL scheme - see the **Open questions** section below |
| User-selectable chart line-density presets | `CANDIDATE` | Adapt the chart to symbol volatility and screen size | First resolve P1.2's red-level threshold and clutter budget; preferences stay display-only |
| Read-only mobile/web dashboard beyond the text digest | `CANDIDATE` | Richer Away review | Define authentication, hosting, freshness, and zero-write boundary after P5.3 |
| Self-hosted ntfy deployment | `CANDIDATE` | More control over notification privacy/availability | Decide operational burden, TLS, backups, and phone reachability; hosted ntfy already works |
| macOS equivalents for Windows scheduled jobs | `CANDIDATE` | Full unattended parity on a Mac | Only if macOS becomes an unattended host; preserve one-main ownership |
| Broader US universe for the M5 strength board | `CANDIDATE` | Closer TC2000 parity | Trader chose the existing ~1,500 universe on 2026-08-15; widen only after the R2 board proves itself and a data/pacing budget is agreed |

## Candidate research and data integrations

| Idea | Status | Value | Prerequisite or decision needed |
|---|---|---|---|
| Capture DYNAMIC and EOD session-VWAP variants in the warehouse | `CANDIDATE` | Compare all champion VWAP interpretations | A registered consumer/question must require them; version the algorithm and keep STANDARD intact |
| Add tier-2 anchor families: catalyst/gap bars, confirmed pivots, and period opens | `CANDIDATE` | Broader AVWAP research | A registered study after the P3.2 pilot; point-in-time anchor availability required |
| Optional completed-bar tracker preview lane | `CANDIDATE` | Earlier visibility after a missed final scan | Trader must request it; it stays labeled preview and cannot write confirmed evidence |
| Additional external universe feeds | `CANDIDATE` | Broader candidate replenishment | Measure incremental source value in P5.5; never auto-remove manual names |
| Options-chain/theta history in the research lake | `CANDIDATE` | Evaluate theta selection and realized support quality | Define data rights, IB pacing budget, schema, and point-in-time quote availability |
| News/catalyst archive with point-in-time availability | `CANDIDATE` | Study catalyst context without hindsight | Choose licensed sources and preserve published/observed timestamps |

## Triggered-later ideas

| Idea | Status | Revisit trigger |
|---|---|---|
| Larger/newer local LLM tiers | `TRIGGERED_LATER` | Model quality materially improves within the 17.4 GiB Vulkan heap, or hardware changes |
| Tree models, survival models, predictive distributions, OOD distance, and calibrated top-K utility | `TRIGGERED_LATER` | Roughly two years of trustworthy corpus plus stable holdouts and prediction ledger |
| FDR/gatekeeping/sequential multiple-testing machinery | `TRIGGERED_LATER` | More than about 100 simultaneous variants, continuous review replacing fixed cadence, or a second researcher |
| Change-point/decay adaptation | `TRIGGERED_LATER` | Durable evidence shows the fixed recent-vs-durable estimator misses meaningful regime shifts |
| Research-lake capacity policy change | `TRIGGERED_LATER` | Lake exceeds 250 GB or backup/restore duration becomes operationally unacceptable |
| Reintroduce immutable bundle import from a second machine | `TRIGGERED_LATER` | The trader deliberately restores a second data-collection role and approves a new ownership/client-ID design |
| SPY wake alarm covering fast intraday reversals | `TRIGGERED_LATER` | After the first live EVENING week with the R1 ±1% alarm |

## Retired and prohibited ideas

| Idea | Status | Reason |
|---|---|---|
| Desk Link satellite/control topology | `RETIRED` | Single always-on main desk replaced it on 2026-08-08 |
| Separate mini-PC live scanner/writer | `RETIRED` | The 8845HS main desk is the sole scan host and writer |
| AI-generated live score, gate, watchlist, alert, or mode changes | `PERMANENT_NO` | AI is one-way advisory; promotion requires deterministic evidence and approval |
| Review-policy suppression or automatic muting | `PERMANENT_NO` | `review_policy.json` ranks/annotates only and has no suppression field |
| Automatic removal of user-entered names | `PERMANENT_NO` | Violates the manual-name invariant |
| Forming-bar confirmation | `PERMANENT_NO` | State transitions use completed bars only |
| Shared mutable home-folder/NAS database | `PERMANENT_NO` | Violates storage and single-writer decisions |
| Order execution or routing | `PERMANENT_NO` | Permanently outside TradingBotV3's product boundary |
| Hiding demoted swing rows from the report | `PERMANENT_NO` | Trader rule 2026-08-15: demote + label, never hide; no suppression anywhere in the quality chain |

## How to promote a wishlist item

Record the trader's decision in one small documentation packet:

1. change the item to `ROADMAP` here;
2. insert it into the correct `plan.md` phase with dependencies and an exit gate;
3. update `CURRENT_CHECKPOINT.md` only if it becomes the active item;
4. create or update a detailed specification only when the roadmap entry needs more
   contract detail;
5. add that specification to `docs/README.md`;
6. do not claim implementation in `CHANGELOG.md` until code or an operational change
   actually exists.

## Trader-entered ideas — 2026-08-14 (triaged and promoted 2026-08-15)

The trader's raw text — including the exact TC2000 formulas — is preserved verbatim
in Git history (commit `994f575`) and carried into the R2/R5 specifications. Every
item below was explicitly promoted by the trader on 2026-08-15 into `plan.md`
Phase 0.5; specs live under `docs/` (see `docs/README.md`).

| # | Idea | Status | Where it landed |
|---|---|---|---|
| 1 | "Not today" on an automatic M5 Focus pick removes that M5 entry | `ROADMAP` | 0.5 R2 — scoped removal, made legal by a new auto-pick provenance sidecar |
| 2 | Place symbol in Focus + mark "I like the stock" from the Alert screen | `ROADMAP` | 0.5 R4 — Add-to-Focus already exists there; LIKE capture is added |
| 3 | AI jobs says "no arguments" at boot | `ROADMAP` | 0.5 R6a — a routine scheduled-task log line (`run_ai_jobs.ps1`); reword + a read-only Health row |
| 4 | Auto journal function | `BUILT` | Nightly `journal_import` slot **built 2026-08-15** as the first JobSlot in the `ai_jobs` slate (R7 §6); five live runs owed. P3.5 commentary journal unchanged and still ROADMAP |
| 5 | Chart Review functions on every chart | `ROADMAP` | 0.5 R4 — CaptureRail embedded on every chart surface |
| 6 | Early-morning D1 gap looks inaccurately large; labeled Y axis | `ROADMAP` | 0.5 R4 — axis labels already exist; real cause is a thin Yahoo forming-bar fallback, fixed by source honesty |
| 7 | Existing price alerts visible on charts | `ROADMAP` | 0.5 R4 — painted alerts/watches as a toggleable levels family |
| 8 | Obvious "already checked today" marking | `ROADMAP` | 0.5 R3+R4 — derived from recorded decisions only (trader choice 2026-08-15) |
| 9 | Auto M5 picks must be above yday HOD and VWAP; evict from queue on fallback | `ROADMAP` | 0.5 R2 — the prev-day half already existed; the VWAP gate, re-checks, and eviction are new |
| 10 | "Any bounce" button across D1/session/H1 levels | `ROADMAP` | 0.5 R5 — AnyBounceWatch; "previous AVWAP" = the prior anchor's VWAP line (new D1 output, trader 2026-08-15) |
| 11 | Structured dislike-reason feedback from the desk | `ROADMAP` | 0.5 R3 — shared veto vocabulary, mechanically counted by the review-learning scoreboard |
| 12 | Master AVWAP setups totally change after the close — investigate | `ROADMAP` | 0.5 R3 — investigation COMPLETE: the live scan scores today's forming D1 bar and the tracker double-writes in the final hour; the full honesty bundle is the authorized fix |
| 13 | Auto-mode semantics clarification (DESK/AWAY/EVENING) | `ROADMAP` | 0.5 R1 — the mode matrix, incl. the EVENING SPY ±1% wake alarm |
| 14 | Auto scanning only 06:30–14:00 PT; quiet boots outside | `ROADMAP` | 0.5 R1 — one fail-open quiet-hours gate; manual scans always allowed |
| 15 | Get rid of shared scan (Google Drive is gone) | `ROADMAP` | 0.5 R1 — the flag is a proven no-op across ~13 files; removal plus stale-Drive string cleanup |
| 16 | TC2000 relative-strength M5 scanner + filters | `ROADMAP` | 0.5 R2 — new pure module over `universe_all.txt` via batched yfinance; existing universe per trader choice |
| 17 | HA reversal + SMI cross + LRSI confluence signal | `ROADMAP` | 0.5 R5 — new indicator modules + Focus-scoped confluence alert |
| 18 | First-candle ORB candidates (gap-up HOD on candle one) | `ROADMAP` | 0.5 R5 — distinct from the existing delayed-ORB detectors |
| 19 | LRSI crosses as their own M5 alert type | `ROADMAP` | 0.5 R5 |
| 20 | Make the program less likely to bog down/crash/overload | `ROADMAP` | 0.5 R6b/c — evidence-ledger rotation (247 MB `technical_integrity_events.jsonl`), stall-watchdog diagnostic week |

## Trader-entered ideas — 2026-08-15 (promoted same day into Phase 0.5 R7/R8)

| # | Idea | Status | Where it landed |
|---|---|---|---|
| 21 | Journal misses trades / trades stuck open — tax-grade completeness from both brokers | `ROADMAP` | 0.5 R7 — Flex-primary IBKR, Questrade activities, coverage ledger + nightly self-heal, position reconciliation, identity fixes (`docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`) |
| 22 | Journal tab like TradesViz/TraderSync: fast edits/notes, tagging, setup performance, walk-away | `ROADMAP` | 0.5 R7 — sub-tabs, corrections with audit trail, R-multiples with alert prefill, pyqtgraph analytics |
| 23 | Per-account P&L with selectable accounts (tax-free vs taxable) + full commission/fee accounting | `ROADMAP` | 0.5 R7 — account tree grouped by tax status, never silently blended; Fees tab + export; CAD tax totals via booked BoC rates |
| 24 | Weekend Prep tab: review the week, focus picks, walk-away, find strongest/weakest H1/D1/Monthly, journal tag review | `BUILT` | 0.5 R8 **built 2026-08-15** — guided 5-step routine, weekend strength boards on the R2 formula, week-ahead prep adoption (`docs/WEEKEND_PREP_PLAN.md`). One-real-weekend live proof owed |

Trader answers that shaped R7/R8 (2026-08-15): journal first, weekend prep second;
both brokers tax-grade; nightly auto-import + self-heal approved; native currency +
CAD tax totals (BoC trade-date rate); R-multiples with alert prefill; charts +
tables; guided routine over dashboard; one strength formula across H1/D1/Monthly;
auto-tag review as the only v1 journal hook in Weekend Prep; adopt to swing Focus +
watchlist; existing ~1,500 universe; spec now, code after the R2→main merge.

Trader answers that shaped the designs (2026-08-15): demote + label, never hide;
v1 extension rules = ATR-distance-from-EMA + outside-AVWAP-bands (the S/R-headroom
rule stays staged, its data already attached); the existing ~1,500-symbol universe
for the strength board; build order R1 → R2 first, R3–R6 behind; the full pre-close
honesty bundle; "previous AVWAP" = the prior anchor's VWAP line itself;
checked-today = recorded decisions only; "Not today" removes just that M5 entry.

## Open questions — items that need one trader decision before they can be built

Merged from `docs/WISHLIST_OPEN_QUESTIONS.md` on 2026-09-05 (repo cleanup); the text below is that file's, unchanged except for heading depth.

Status: **active** — one question per item, each blocking a build that is
otherwise ready. Created 2026-08-18 under the trader's integration redirect,
which asked for every implementable `WISHLIST.md` item to be built and for
anything "too vague to build without a trading judgment" to get a spec stub
stating the open question instead of a guess.

This section is not a roadmap and nothing here is authorized: an item leaves this section
by the trader answering its question, at which point it follows the normal
promotion path into `plan.md`.

**Why a stub rather than a best guess.** Each item below has exactly one
unanswered question whose plausible answers lead to *different code*, not
different polish — a different storage location, a different data budget, a
different failure posture. Building the wrong branch and calling it a default
would hide the decision inside an implementation, which is the thing the
ask-first rule exists to prevent.

---

### Built instead of stubbed (2026-08-18)

| Item | What landed |
|---|---|
| Deep-link a symbol/timeframe into TradingView or TC2000 | `scripts/external_chart_links.py` plus an **Open in TradingView** button on the arm bar, so every chart surface that carries the bar inherits it. The URL is a machine-local setting (`external_chart_url_template`), the symbol is validated before a URL is built, and a refused open is reported rather than swallowed. **TC2000 is deliberately not wired**: it is a desktop app whose documented automation surface is its own scripting layer, not a URL scheme, and a `tc2000://` link that silently does nothing would be worse than the honest gap — the template setting is the seam for it the day the trader confirms what their install answers to. |

---

### User-experience items

#### Voice dictation for the live commentary journal

**Open question — local or cloud speech, and what happens to a bad
transcription?** Local (whisper.cpp on the 8845HS) keeps every word on the desk
and costs GPU time during a session; cloud is more accurate and sends the
trader's live commentary to a third party. The correction workflow follows from
that choice: a local model needs an edit-before-commit step, a cloud one could
commit and correct after.

Blocked also by ordering: `plan.md` P3.5 owns the commentary journal itself,
and there is nothing to dictate into yet.

#### User-selectable chart line-density presets

**Open question — what does "too many lines" mean on the trader's screen?**
The prerequisite recorded in `WISHLIST.md` is P1.2's red-level threshold and
clutter budget, and that is a desk-evidence decision, not a preference toggle:
presets built before it would encode a guess about which levels matter, and the
trader would then be choosing between three wrong densities.

#### Read-only mobile/web dashboard beyond the text digest

**Open question — who may read it, and from where?** A phone-reachable page
showing positions and candidates is an authentication and hosting decision
before it is a UI one, and the answer changes the whole build (a LAN-only page
behind the router is a different system from an internet-reachable one with
accounts). `plan.md` P5.3's one-snapshot work is the natural prerequisite.

#### Self-hosted ntfy deployment

**Open question — is the operational burden worth it?** Hosted ntfy already
works and costs nothing to run. Self-hosting buys privacy and control and costs
TLS certificates, a reachable endpoint, backups, and a new way for the phone to
go silent on a Sunday. This is an operations judgment, not a code one; the
sender is already a thin seam, so the build is small once it is wanted.

#### macOS equivalents for the Windows scheduled jobs

**Open question — will a Mac ever be the unattended host?** The wishlist entry
answers this conditionally ("only if macOS becomes an unattended host"), and
today the 8845HS is the sole always-on machine and sole writer. Building launchd
equivalents now would create a second scheduling surface that nothing runs and
nobody tests.

#### Broader US universe for the M5 strength board

**Explicitly gated, not vague.** The trader chose the existing ~1,500-name
universe on 2026-08-15, and the entry says to widen it only after the R2 board
proves itself and a data/pacing budget is agreed. That is a live gate; it is
listed here only so the reason it was skipped is recorded.

---

### Research and data items

Every entry under "Candidate research and data integrations" shares one shape:
each needs a **registered consumer** before it may be captured. That is the
locked warehouse plan's own rule (`docs/ULTIMATE_SETUP_DATABASE_PLAN.md`) —
capture is justified by a question someone will actually ask, because an
append-only lake makes "collect it in case" a permanent cost.

| Item | The one question |
|---|---|
| DYNAMIC and EOD session-VWAP variants in the warehouse | Which registered study needs the variants? Capturing all three triples the VWAP surface and STANDARD must stay untouched either way. |
| Tier-2 anchor families (catalyst/gap bars, confirmed pivots, period opens) | Same: which study, and is the anchor's point-in-time availability provable at the simulated decision time? |
| Optional completed-bar tracker preview lane | The trader must ask for it. It is a labeled preview lane that can never write confirmed evidence, so its value is entirely about whether the trader wants an earlier, weaker read after a missed final scan. |
| Additional external universe feeds | What is the incremental value over the current sources (P5.5 measures it), and what is the pacing budget? Manual names can never be auto-removed regardless. |
| Options-chain / theta history | Data rights, IB pacing budget, and whether point-in-time quotes are actually retrievable. Any one of those can make the capture impossible rather than merely expensive. |
| News/catalyst archive with point-in-time availability | Which licensed source, and can published-vs-observed timestamps both be preserved? Without both, hindsight leaks into every study that reads it. |

---

### Not stubbed, deliberately

`TRIGGERED_LATER` items already carry explicit revisit triggers and
`PERMANENT_NO` items are settled product boundaries. Neither belongs here:
the first are waiting on a measurable event, and the second are not questions.
