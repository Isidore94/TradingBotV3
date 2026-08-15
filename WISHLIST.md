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

AI jobs says "no arguments" when booting up. useless right now. need to assess what can be improved here or if there are any easy wins here
auto journal function 
the chart review tab is a big thing I want to integrate but it never gets used because I use trading desk most of the time
sometimes the D1s early in the morning show a inaccurately large gap. i wonder if a labeled Y axis with the price levels would be useful 
it would be useful if nay price alerts appeared on charts so I can what is already set from before
anytime I bring up a chart from master avwap setups or the RS?RW board or anywhere it would be nice if it had all the functions of chart review. it would also be nice if it made it very obvious I have already checked that chart today 
ANY auto m5 focus pick needs to be above the previous day high for longs and honestly above vwap as well on hte m5. when backfilling if the stock flals below vwap or yday HOD it should just be removed while its in queue, similiar for shorts we need below yesterdays LOD and below vwap
we honestly need a "any bounce" button. for longs and shorts finds any bounce be it D1 off 1stdev or avwape or prevous avwape or previois 1stdev or the 15ema or the 21ema or the 15ema on an H1 basis. the idea is I will find stocks breaking out and if I want a pullback I want to say "find them all" and if I still dislike it when that laert fires thenI can set it again. most stocks will either get a price alert level or a pullback alert. this is a lot of logic to track but it is useful
similarly I need to also be able to give feedback as to why I dislike certain stocks. chart review again has hit it out of hte park we can expand on it but we need these features integrated 
ive noticed sometimes master avwap setups will TOTALLY change after the close. investigate why I want these great setups before the close so I can actuate on them.
need some more clarificatin for the auto modes. Auto desk is when I am at the desk. dontn eed phone alerts. do want auto m5 focus picks and trades served to me live. auto away is when I am at work, theres no need to populate trades in the GUI if theres any focus picks that hit a level or any alerts that go off just queue them all and keep them ready for when I get home. dont bother showing any M5 focus picks I wont get to them. send me updates of the best master avwap setups to my phones in addition to any price alerts from the research tab to my phone. when I get home I will go through the GUI and do all the work to update the focus picks or deal with any alerts. Auto evening is for when we have finisihed trading for the day and I go work an evening sift that gets me home very late. we dont need any new scans or anything we just need M5 focus picks queued and phone alerts to go off. in particular we want phone alerts for if SPY is up or down more than 1% and we want that alert to be sent every 5 minutes from the market open to whenever I chage the auto mode to desk to signify im awake. the main focus of this mode is to get the days trades ready for me in addition to waking me up if a stock makes a big move (from the resaerch tab alerts) or if the market makes a big move and I cannot afford to sleep in
the bot should only be scanning in auto mode during market hours 0630 am PST or 0930 am EST and up to 1 hour after the close (1400PST or 1700 EST). if I boot up the bot outside those hours it souldnt start getting ATRs or doing anything. outside those hours its quiet unless I MANUALL run a local scan
get rid of shared scan. theres no more google drive with watchlists its all local now. 
Something I find myself doing in TC2000 that the bot doesnt have is I have a formula for relative strength (((((C11/O11)-1)*100)  +(((C10/O10)-1)*100)  +(((C9/O9)-1)*100)  +(((C8/O8)-1)*100)  +(((C7/O7)-1)*100)  +(((C6/O6)-1)*100) + (((C5/O5)-1)*100)  +(((C4/O4)-1)*100) + (((C3/O3)-1)*100) + (((C2/O2)-1)*100) + (((C1/O1)-1)*100) + (((C/O)-1)*100))/12*(((C+C50)/2)/ATR50)) and then I sort by the top 25% of stocks in the US and it basically finds me the strongest picks in the market on an M5 basis. I would appreciate if fixed up our RS/RW board to be a bit more useful and to mimic this scanner. I basically look for this along with stocks above vwap, with more than 1m avg volume per day over a 20 day period, a price above 5, a market cap above 1B, has options, is above yesterdays HOD and is above the 15ema. I abuse this scan for longs and invert for shorts and Im always able to find great strength early in the day. i always dump these into my m5 focus picks. there tends to be about 20-40 per long and short side. some way to get this function working would be awesome
id like to add HA reversals with a recent or concrurent SMI cross where SM1 is below SM2 and both are below 0 and then SM1 crosses above SM2 here is how I code it in TC2000 XUP(XAVG(XAVG(C - (MAXH5 + MINL5) / 2, 5), 20) / XAVG(XAVG(MAXH5 - MINL5, 5), 20),
    XAVG(XAVG(XAVG(C - (MAXH5 + MINL5) / 2, 5), 20) / XAVG(XAVG(MAXH5 - MINL5, 5), 20), 6))
and XAVG(XAVG(C - (MAXH5 + MINL5) / 2, 5), 20) > 0 
I also accompany the above with an LRSI cross herse how I code that in tc2000 (ABS(C >= XAVGC9.1) * (XAVGC9 - XAVGC9.1) + ABS(C1 >= XAVGC9.2) * (XAVGC9.1 - XAVGC9.2) + ABS(C2 >= XAVGC9.3) * (XAVGC9.2 - XAVGC9.3) + ABS(C3 >= XAVGC9.4) * (XAVGC9.3 - XAVGC9.4))/ (ABS(XAVGC9 - XAVGC9.1) + ABS(XAVGC9.1 - XAVGC9.2) + ABS(XAVGC9.2 - XAVGC9.3) + ABS(XAVGC9.3 - XAVGC9.4)+ .0000001) * 100 
so im looking for HA rev then an LRSI reversal (ideally below 20 the then above 20 but below 50 and then above 50 works) as well as an SMI reversal. its a strong signal but hard to caculate so should only be used for M5 focus picks. give lots of leeway they may not happen at the same time but should occur within 3-4 candles of each other on an M% basis, 
Ensure the auto desk mode can find M5 ORB candidates. today UMAC was excellent. gap up over compression made a HUGE m5 first candle with a big wick, pullbed back then eventually got above vwap then broke above the days HOD. I think stocks that make a HOD on the first candle on a gap up are candidates for ORB trades. id like an LRSI pullback then an alert on a new HOD automatically. I should also get an alert on LRSI above 50 as well. just so i can put it on my rader. 
while we are at it we should add LRSI crosses as its own alert type on the M5. in general HA revw ith LRSI and SMI is the storngest but sometimes just LRSI will give time for hte stock to reset before I need to check it again. 
consider ways to make the program lessl ikely to get bogged down/crash/overloaded. important to spread out resources we run this on powerful systems. this one is on a ryzen 7 8845HS with 32gb of DDR5 ram. 
