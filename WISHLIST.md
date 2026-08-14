# TradingBotV3 integration wishlist

Last reconciled: **2026-08-10**

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

## Candidate user-experience integrations

| Idea | Status | Value | Prerequisite or decision needed |
|---|---|---|---|
| Voice dictation for the live commentary journal | `CANDIDATE` | Faster capture while watching charts | Choose local vs cloud speech, privacy, correction workflow, and storage format after P3.5 |
| Deep-link a symbol/timeframe into TradingView or TC2000 | `CANDIDATE` | Faster transition to external deep TA | Confirm supported URL/application schemes and failure behavior; no scraping or browser automation dependency |
| User-selectable chart line-density presets | `CANDIDATE` | Adapt the chart to symbol volatility and screen size | First resolve P1.2's red-level threshold and clutter budget; preferences stay display-only |
| Read-only mobile/web dashboard beyond the text digest | `CANDIDATE` | Richer Away review | Define authentication, hosting, freshness, and zero-write boundary after P5.3 |
| Self-hosted ntfy deployment | `CANDIDATE` | More control over notification privacy/availability | Decide operational burden, TLS, backups, and phone reachability; hosted ntfy already works |
| macOS equivalents for Windows scheduled jobs | `CANDIDATE` | Full unattended parity on a Mac | Only if macOS becomes an unattended host; preserve one-main ownership |

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

## Trader-entered ideas — 2026-08-14

1. When we choose **Not today** on an automatic M5 Focus pick, remove it from the
   M5 Focus watchlist.
2. Add a way to place a symbol in Focus from the Alert screen, and a way to mark
   **I like the stock** there. Chart Review has many of these tools, but the trader
   primarily uses the Trading Desk.
