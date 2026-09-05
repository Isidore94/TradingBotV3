# Project process review — 2026-09-04

Assessment of `main` at `f641421`. Advisory review, not a new roadmap or authorization to change trading behavior. Governing priorities: decision 0016; `plan.md` §§5–7; current V4, Phase 0.15 S2/F3 and gate #59. Source, current inventory, recent Git history, relevant specifications and bounded runtime evidence were reviewed. Archived history was not loaded wholesale. No full lake rebuild, market-data request, GUI restart or full regression run was performed.

## Conclusion

The architecture supports both goals well: finding names to trade now and collecting evidence to find better names later. The highest-value next work is **making the measurements and summaries trustworthy, then using them on the Trading Desk**. More detectors, larger model contexts and a broad rewrite would not address the main gaps found here.

Recent performance work is real. The warehouse child process, persistent tee high-water marks, shared calendar cache, coalesced UI refreshes and reduced DESK scan cadence should be preserved. Smooth chart interaction and timely scanning are separate acceptance criteria; the latter still needs attention.

## What already accomplishes the goals

- **Swing discovery:** the D1 scanner supplies earnings-anchored AVWAP setups, current/previous anchors, structural levels, candidate buckets, tracker outcomes and Expected R. The phone digest now compares favorite and near candidates using family Wilson bounds at a declared horizon. The old favorite-only complaint is already addressed (`autopilot_core.py:3460,3503`). Yahoo daily bars are an intentional local pin, not proof of failed IB service.
- **Intraday discovery:** completed M5 bars drive BounceBot; the TC2000-style strength board feeds eligible DESK Focus picks through the existing gate. Chart review, armed watches and recorded trader decisions share the desk. Held-level/MFE summaries already reach the tracker and alert suffix. The champions remain separate from these advisory measures.
- **Research:** immutable lake parts, manifest resolution, duplicate repair, versioned recipes, registered trials, costs, ambiguity, coverage and episode counts exist. The anchor bridge is built; historical band completeness and recomputed results still require gate #59. A completed repair of duplicated M5 bars is not proof that missing D1 bands were repaired too.
- **Learning from the trader:** quick likes, claimed likes, vetoes, passes, rejections, journal links and after-like studies are built. A like is evidence of preference, not proof of edge or an instruction to take a position.
- **Two-tier AI:** deterministic digests, bounded narration views, source references, resumable ticker briefs and compact publication already exist. The right improvement is to strengthen their contracts and navigation, not ask the frontier model to reread the lake.

## Highest-priority findings

### 1. Fix what “held” means before giving it ranking influence

`held_run_score.Episode.held` is currently `not broke_early` (`scripts/held_run_score.py:198`). A registered event with no follow-up is therefore treated as held. A pure in-memory probe returned `held=True, mfe=0.0` for that case. The producer writes registration immediately (`bounce_bot_lib/legacy.py:4885`), so absence of follow-up is a real input shape.

The elapsed time attached to a later update is also not necessarily the first stop-hit time: the producer checks bars since entry while the consumer compares the update's elapsed time against 30 minutes (`legacy.py:5120,5126,5139`; `held_run_score.py:323`). A late first observation can misclassify an early break.

**Recommendation:** separately count measured-held, measured-broken, pending and unmeasured episodes. Establish the first break time or state that it is unknown. Preserve every event and display coverage beside the headline. Characterize current behavior first; prove that missing follow-up never counts as confirmed holding. The number of live rows affected was not measured. This is a prerequisite recommendation for V4, not authorization to change its score.

### 2. Make the D1/M5 overlap match the actual thesis

The overlap join uses session and symbol, discarding side and intraday knowledge time (`held_run_score.py:471,489,623`). An in-memory probe showed a short D1 favorite marking a long M5 event as having a D1 setup. A missing snapshot becomes False rather than unknown (`:610`). This cannot distinguish an aligned swing from an opposing swing or one discovered later that day.

**Recommendation:** retain side and known-at time, and distinguish aligned/opposed/unknown. Define which category may receive priority before implementing it. A retrospective same-day join may be descriptive, but is not evidence that the supporting setup was known when the alert fired. Test opposite sides, absent snapshots and a D1 setup learned after the M5 alert.

Also align “lately”: held-run currently selects the last 20 dates present in its file (`:341,440`), while the swing path uses exchange sessions. Sparse data can turn 20 observed dates into a much older window. Use the shared exchange-session cutoff and report gaps; the current behavior is deliberately tested in `tests/test_v1_held_run_score.py:214` and needs a contract change.

### 3. Finish the band repair with eligibility and vintage checks

The anchor bridge fixes the missing input feed, but the normal build writes daily features for its selected day (`research_warehouse/cli.py:716`). Outcome recomputation reads the feature row for the original trigger session (`:628`). Running only outcome recomputation cannot populate old missing feature rows.

**Recommendation:** verify the full chain: cached anchors → appended CSV → bronze → anchor instances → required historical daily features → recomputed outcomes → new fact pack. Report required-band coverage and valid directional geometry by recipe. Keep incomplete/fallback paths separately labeled; do not present them as fully observed house-management trials merely because n clears a floor.

Preserve point-in-time provenance. Anchor instances carry `system_from` (`features.py:378`), but `anchor_dates_by_symbol` checks anchor date, not that knowledge timestamp (`cli.py:264`). Reconstructed historical data must not silently become evidence that the desk knew the anchor then. State observed versus reconstructed eligibility and test it before using repaired history for promotion. No historical lake write was made in this review.

### 4. Correct misleading research conclusions before the frontier model reads them

The two open investigation documents contain useful counts but several conclusions do not follow from the code:

- The like writer stores **`match_basis` inside payload**, whereas both assessment readers ask for `basis` (`like_links.py:89`; `docs/analysis/scripts/lake_assessment.py:495`; `lake_likes_and_details.py:34`). A bounded live read found **84 bronze versions, 77 distinct event IDs; latest by `observed_at`: 41 matched `any_family`, 36 `none`**. Zero linked likes is false. These are a later snapshot, not a reconstruction of the earlier 74 rows.
- A missing-target fallback cannot reach TARGETED, but it **can make money at expiry**. Calling the existing `_walk_plain` with entry 100, stop distance 2, close 103, no target and a one-session expiry returned `EXPIRED`, `gross_r=1.5`. Thus “guaranteed loss” and “wins structurally impossible” are incorrect. The historical 0/257 resolved result remains a reported observation, not proof that all open rows must lose.
- The `m5close_*` study enters after a D1 signal and can continue for 18 sessions (`outcomes.py:202,935,1079`). It is not the live intraday bounce population. Its negative returns cannot establish that live day-trade alerts fail, nor does favorable excursion alone prove an executable edge.
- A positive control win rate alone does not establish edge, and MFE alone does not identify exits as the cause of negative returns. Inputs, entry selection, costs, stop distance, ambiguity, censoring and a matched baseline all matter. The earlier probability claim about 0/257 is also invalid and should not be reused.

**Recommendation:** reuse production payload readers and outcome definitions in audit scripts. Validate schema fields before rendering a report; an unknown field must produce an audit error rather than a confident business conclusion. Historical reports now carry explicit correction notices. The audit scripts and their saved JSON were not rerun or repaired in this review.

### 5. Make the local AI summarize facts without changing their meaning

The current local morning file for session September 3 says “Briefed 152 of 152,” including membership-only cases. It also calls BULL a held long position while citing watchlist membership (`C:\TradingBotData\ai_morning_brief.txt`, around lines 14–35). The file was 48,990 bytes when inspected. The source validator checks shape and source IDs, not whether those IDs support a position claim or a stated number (`scripts/ai_summary.py:2235,2288,2308`).

**Recommendation:** supply typed deterministic fields: watchlist membership, confirmed journal position, scanner finding and unavailable evidence. Permit a position claim only from the appropriate position source. Require numeric claims to reference an exact metric/cell, horizon and denominator. Publish analyzed / membership-only / failed counts instead of one success-looking total (`briefs.py:761`). Unsupported prose should be rejected or omitted while verified facts remain available. This improves both local output and the frontier model's input without increasing inference work.

### 6. Protect the deterministic overnight work and its gates

Summary and ticker narration precede several deterministic outputs in the runner (`ai_jobs/runner.py:514,534,554,639,662`). Reservations can cause later jobs to skip (`:323`). That is a code-supported starvation risk, not a newly measured nightly failure.

`clean_digest_sessions` counts distinct pack dates rather than consecutive clean exchange sessions (`digest.py:1156`); `digest_gate_state` uses that count (`:1178`), and journal enrichment consumes the Boolean (`enrichment.py:155`). The governing ten-session collection condition and separate spot audits are stronger than this date count.

**Recommendation:** enforce clean/consecutive collection and retain separate recorded audit approval. Consider a dependency-aware deterministic stage before optional narration, with deadlines, resumable work and last-good publication. Slot order is an explicit existing contract, so change it through a decision, not an incidental reorder. Keep inference outside the trading window and keep automatic frontier synthesis unbuilt unless separately authorized.

### 7. Keep the frontier handoff small and auditable

Extend the existing digest publication with a compact entry index, not another raw-data export. It should name the latest complete session, code/recipe/input versions, corrected or superseded packs, coverage and failures, and a few supported changes from the prior comparison window. Each finding should link to its deterministic cell and the full artifact.

Keep intraday held-level/MFE, swing win rates, preference observations and journal execution results separate. Show independent episodes beside recipe rows. List pending experiments and their frozen windows without ranking immature cells. A frontier model should open a detailed ticker brief only for a stated question. This prevents repeated model summaries from laundering an unsupported statement into a “fact.”

## Performance: preserve the gains, target the remaining measured work

Read-only samples from today's restarted desk showed the tee at **at most 0.8% of one core when present in the top-thread samples** over 11:25–14:10 PT. That supports the recent fix; absence from a top-thread list is not a zero measurement. The same 166 samples contained 60 where `run_strategy` used at least 50% of a core.

Today's logged scan preambles still took **563.6 s, 597.7 s and 1,125 s**. The long case attributed about 920 s to Focus work; two others attributed about 212–221 s to H1 work. These are preamble timings, not measurements of every alert's latency. Instrument bar-close → evaluated → shown age per engine and source before changing cadence or detection. First inspect repeated Focus work and H1 work on unchanged completed bars. Cache by input version only where golden results and lifecycle behavior remain identical. A detector child process is a possible later isolation step, not a prerequisite for this review.

A specific remaining UI stall was recorded at **13:00:44 PT: 15,739 ms** in `_score_todays_picks`. It materializes CSV rows on the GUI thread and filters afterward (`scripts/ui/services/autopilot_service.py:1500,1552,1556`). The two runtime CSVs were about 335 MB and 308 MB. Both the regular tick and wrap-up worker can call it (`:527,1473`); the completion marker is set before scoring (`:1496`). Move this existing calculation to one owned worker and narrow its reads; return only the small result for painting. Test equivalent output, duplicate triggers producing one result, failure permitting retry while preserving last-good output, and no synchronous file read in the GUI callback.

Measurement sources: `%LOCALAPPDATA%\TradingBotV3\diagnostics\thread_cpu.jsonl`, `diagnostics\ui_stalls.jsonl`, and `logs\trading_bot.log`. The watchdog is capped and thread reporting is sampled; neither establishes the absence of all other stalls. H1 recomputes SPY bars inside its per-symbol loop (`bounce_bot_lib/legacy.py:8851`), a candidate for shared per-sweep work, but its share of the H1 elapsed time was not measured. Separate request waits from calculation before choosing the fix.

The tracker SQLite mirror is already built. Verify the five-save parity gate #57 before moving readers individually; a wholesale storage migration would unnecessarily risk the recent gains. Keep worker CPU, GUI stalls, memory and scan freshness in the same acceptance record. Include a busy open, a full post-scan build, chart navigation and end-of-day scoring; test at late-month data volume as well as early-month volume.

## How to evaluate better picks

After the measurement fixes, V4's existing “Working lately” surface and display-only priority switch offer the clearest payoff. The switch should compare the **same captured candidate set** with the current order, preserve every visible row and armed alert, and be reversible. Freeze the question and forward window before reading results.

Measure top-of-list useful names per review budget, time spent reviewing, freshness, and coverage. For day trades use same-session held-level outcomes and MFE with adverse excursion/time-to-move; for swings use the declared tracker win verdict and uncertainty, with net returns and loss size alongside it. Compare sides separately and matched populations; multiple recipes for one occurrence do not create independent trades. Likes describe personal fit, while outcomes test objective usefulness.

Finish the already-scoped Setup Types win-rate export and AWAY Recap using these shared facts. Do not add more screens or setup grids simply to fill empty output. More warehouse families and new rankers belong to their existing later phases and promotion gates.

## Verification and scope limits

Three read-only recon reviews covered discovery, AI and performance. Direct inspection confirmed the band fallback, historical-feature dependency and audit-field mismatch. Pure in-memory probes reproduced missing-follow-up holding, opposite-side overlap and profitable no-target expiry. The tiny like-link dataset was counted through manifest-resolved reads; no full fresh lake audit was performed. Runtime evidence is a sample, not a completed live gate.

Only review/control documentation changed. Runtime code, tests, historical analysis scripts/JSON, settings, live stores, champions and policies are unchanged. Relative links and identical CLAUDE/AGENTS copies were checked; `git diff --check` passed. The existing recorded full-suite baseline remains **6608 passed, 1 skipped, 72 subtests**; it was not rerun or newly certified here. Suggestions above remain advisory under the existing roadmap and ask-first rules.
