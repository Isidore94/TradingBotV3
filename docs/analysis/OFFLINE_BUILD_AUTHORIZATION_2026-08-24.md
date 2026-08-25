# Offline-build authorization — 2026-08-24 (night)

**Hand-committed and frozen.** Trader decisions from the 2026-08-24 night
conversation, following completion of the R10 slate (10 of 10, `db36459`).
Directive: *"build everything that doesn't directly require testing."* This
record scopes that directive precisely; plan.md still owns build order inside
it.

## 1. The §6.4a digest questions are ANSWERED (trader, 2026-08-24)

The 2026-08-08 decision ("no digest schema may be built or frozen before the
trader answers") is satisfied. Answers, recorded verbatim against
`docs/LOCAL_AI_AUTOMATION_PLAN.md` §6.4a's six open questions:

1. **What counts as winning:** BOTH, side by side — R at scenario close AND
   MFE/MAE reported together, never blended (consistent with R10 ground rule
   12: MFE/MAE are opportunity, close-R is result).
2. **First-class slices:** environment × day-part (the `env_key` R10.A already
   stamps) × side. No setup-family slice in v1; adding it is a v2 decision.
3. **Shadow-engine outputs:** EXCLUDED from the digest — champion facts only.
4. **Retention:** narration files are disposable and regenerable; fact packs
   are permanent record.
5. **Cap:** 16 KB hard cap stands.
6. **Non-sessions:** a weekend/holiday writes an EMPTY fact pack so the gap is
   visible — never a missing file.

Phase 2 (§6.4a fact pack + narration) may now be built. Its live gate — ten
clean digest sessions, trader spot-audit — is unchanged and owed.

## 2. Two recorded reversals, approved

- **R7 true-USD conversion**: the 2026-08-18 "stays deferred" is reversed;
  build it (BoC rates already land nightly per R7 I5; a missing rate still
  renders "unconverted", never 0).
- **LOCAL-AI Phase 3/4 machinery, runs gated**: the journal-enrichment nightly
  pass (advisory fields only, never overwriting trader data) and the
  `review_policy_draft.json` writer may be BUILT now, ahead of their phase
  gates, on the R10.I scaffolding pattern — each refuses to run (or labels its
  output non-authoritative) until its recorded gate passes, drafts only,
  `review_policy.json` untouched. Phase 4's two-week frontier-vs-medium
  side-by-side quality gate is unchanged.

## 3. Wave 1 — authorized to build now (offline, no authority cutovers)

1. Phase 2 digest: deterministic fact pack + medium-tier narration per §6.4a
   with the §1 answers; schema v1 frozen accordingly.
2. Weekly synthesis machinery over the graded cohorts (deterministic
   `evidence_stats` rollup + medium-tier narration over that fact pack only),
   gated: refuses/labels until two weeks of graded rows; **no frontier call —
   still unauthorized**.
3. R8's remaining deferred joins: `human_focus_performance.csv` and
   `pick_feedback.jsonl` in Focus Pick Review; the
   `rrs_group_strength_extremes.csv` stream in Week in Review.
4. R7 true-USD conversion (§2).
5. LOCAL-AI P3/P4 machinery, runs gated (§2).
6. P1.1 hermetic test suite; P1.4 observability depth (benchmark/golden
   fixtures and trends); P1.5 bounded repository hygiene — the Desk
   Link/satellite/mini-PC retirement in an explicit, fully green cleanup
   packet, never mixed with behavior changes.
7. Reconcile plan.md P3.5 against R10.H (the market commentary journal largely
   landed there; P3.5 retains only what R10.H did not build).

## 4. Deliberately NOT in Wave 1, and why

- **P1.2** (D1 line-display thresholds) needs desk evidence; **P1.3** (branch
  adjudication) is trader judgment; **P3.4** is time; §7 promotion work needs
  live windows.
- **Phase 2 of plan.md (P2.1–P2.6: provider repository, point-in-time
  repairs, CandidateRegistry authority, RS integration, Greatness lane) and
  P6.4** are offline-heavy but are AUTHORITY CUTOVERS stacking on a branch
  that already carries ~25 unvalidated packets. Recommendation recorded: run
  the live validation day and merge to `main` first; Wave 2 gets its own
  authorization after that. The trader has not overridden this
  recommendation.
- **Phases 4, 5, 6 (rest), 7**: depend on Phase-2 authorities that do not
  exist yet; their exit gates are live by construction.
- **Frontier calls, nightly model reads of raw streams**: unchanged, not
  authorized.

## 5. Unchanged

plan.md sec 5 invariants; R10 ground rules; every owed live gate (five
mechanics canaries, R10.V scan day, R9's four proofs, fluidity gates, R7
gates 1/3/6, R8's weekend, ten digest sessions once Phase 2 runs); the
Questrade token paste; nothing merges to `main` until a live-session
validation day passes.
