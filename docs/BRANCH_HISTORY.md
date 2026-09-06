# Branch history and the consolidation to `main`

Last reconciled: **2026-09-05**

This file records what each development branch was, and where its work ended up, so
that deleting a merged branch never destroys the only account of what it contained.
It is a provenance record, not a roadmap: what the branches *built* is in
[`CHANGELOG.md`](../CHANGELOG.md), and what is still owed is in
[`plan.md`](../plan.md).

**Dated checkpoint entries and the frozen `analysis/` records still name these
branches**, because that is where the work truthfully happened. Those references were
deliberately left alone rather than rewritten - editing history to match a deletion
would be the more dishonest repair. This file is how a reader resolves a branch name
that no longer exists.

Commit counts below are measured against the pre-consolidation `main`
(`7d85a27`, "Complete the Phase 1 exit gate to Sol's full criteria").

## The 2026-08-26 consolidation

Between 2026-08-04 and 2026-08-25 the Phase 0.5 work ran on a chain of branches
rather than on `main`, because the trader was running unmerged branch code in
production through a scheduled task (`docs/archive/CHECKPOINT_REVIEW_2026-08-08.md`). Each
branch was cut from the previous one and merged forward, so the chain is nested
rather than parallel: **`testing-week-2026-08-24` contained every commit of its
predecessors.**

`main` was a strict ancestor of `testing-week-2026-08-24`, so the consolidation was a
fast-forward. No conflict was possible and no merge resolution was performed.

### Branches that landed

| Branch | Commits | Range | Tip | Disposition |
|---|---|---|---|---|
| `testing-week-2026-08-24` | 354 | 2026-08-04 → 2026-08-25 | `ed277a7` | **The release candidate.** Fast-forwarded onto `main` on 2026-08-26. Kept alive — active GUI-optimization work continues on it |
| `phase05-integration-blitz` | 308 | 2026-08-04 → 2026-08-23 | `1a2fbde` | Contained in `main`. **Cleared for deletion 2026-08-26; deletion still owed** (see below) |
| `testing-week-2026-08-17` | 262 | 2026-08-04 → 2026-08-20 | `170172b` | The previous week's release candidate. All but one commit contained in `testing-week-2026-08-24`; the exception is a doc-reconciliation note superseded by the 2026-08-25 reconciliation. Branch retained for now |
| `phase05-r2-focus-gating-strength-board` | 150 | 2026-08-04 → 2026-08-18 | `a8c696a` | R2 Focus gating and the M5 strength board. Contained in `main`. **Cleared for deletion 2026-08-26; deletion still owed** (see below) |
| `claude/ticker-briefs-hardening-imcm8r` | 94 | 2026-08-04 → 2026-08-11 | `9e0df9e` | Ticker-brief hardening and the first night's measurements. Contained in `main`. **Cleared for deletion 2026-08-26; deletion still owed** (see below) |
| `claude/trade-analysis-opus-prompt-vgg1n8` | 1 | 2026-08-22 | `6c1398f` | One additive document, merged into `main` on 2026-08-26 as `docs/archive/prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md` |

### The deletion itself is owed, and why

The three branches above hold **no commit** that is not on `main`. That was verified
with `git merge-base --is-ancestor` against the post-merge `main` (`226fbac`), not
assumed, so deleting them discards nothing.

**The deletion could not be performed from the cloud session.** Its GitHub credential
pushes fine - `main` and a new branch both went through in the same session - but
refuses ref deletion with `HTTP 403`, and the egress proxy recorded no policy denial,
so the refusal is GitHub-side token scope rather than a blocked host. The GitHub MCP
surface offers `create_branch` and no delete counterpart. Run from the desk:

```
git push origin --delete claude/ticker-briefs-hardening-imcm8r
git push origin --delete phase05-r2-focus-gating-strength-board
git push origin --delete phase05-integration-blitz
```

Re-prove containment first if any time has passed - the check is one command per
branch and it is the whole safety argument:

```
git fetch origin --prune
git merge-base --is-ancestor origin/<branch> origin/main && echo SAFE
```

Update this table when the deletions land.

### Work that did NOT land, and is not lost

| Branch | Commits | Tip | Why it is still open |
|---|---|---|---|
| `claude/alert-center-quality-packet-5btu3w` | 8 | `57fcf47` (2026-08-18) | See below |

**The Alert Center quality packet is unmerged by trader decision (2026-08-26), not by
oversight.** It builds the alert-delivery measurement surface named in
`GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` sec 10.3 and sec 17: `scripts/alert_quality.py`
(951 lines), `scripts/alert_delivery_events.py` (437 lines), a delivery-capture emit
inside `scripts/ui/panels/alert_center_panel.py`, a System Health surface so a dead
emit site is visible, and roughly 1,050 lines of tests. Its own packet document claims
Phases 0–2 landed and green, Phase 3 partly, with Phase 4 as an owed live gate.

Two things must be settled before it can be merged:

1. **It edits alert code.** `scripts/ui/panels/alert_center_panel.py` houses alert
   behavior, so the CLAUDE.md file-scoped ask-first rule governs the merge itself, not
   only any later edit.
2. **It carries a filename collision.** The branch adds its own
   `docs/archive/ALERT_CENTER_QUALITY_PACKET.md` — a live spec for the packet it builds —
   while `main` already carries a *different* file at that exact path: the historical
   P1.6 packet recovered byte-for-byte from `671ee57` and classified as historical
   evidence in `docs/README.md`. Merging without renaming one of them would conflict,
   and resolving the conflict by content would silently destroy one of the two.

Until both are answered the branch stays as it is. Nothing on `main` depends on it.

## The 2026-08-31 evening integration

Two independent lines were merged into `main` on the evening of 2026-08-31 and their
branches deleted. Both were cut from `main` at `50af716`, so they were parallel rather
than nested: the snappiness line fast-forwarded, and theta was a real merge whose only
conflict was in `CURRENT_CHECKPOINT.md` - both sides' dated entries were kept.

### Branches that landed

| Branch | Commits | Tip | What it held | Disposition |
|---|---|---|---|---|
| `claude/desk-snappiness-3` | 14 (its own 5, plus packets 1-2) | `6df2036` | Desk snappiness packet 3: the technical-integrity resolved sidecar so the nightly replay stops streaming 618 MB, the Industry Board's quiet-hours gate and chunked download, three hidden-page timer gates, and eight measured drips. Also the reviewer's fix so the sidecar cannot count one resolved event twice | **Fast-forwarded onto `main` 2026-08-31** (`50af716..6df2036`). Contained; deleted local + origin |
| `claude/desk-snappiness-2` | 4 | `93bbe1b` | Snappiness packet 2: the Alert Center minute tick (M5 bar memo, one D1 level build per symbol, batched prefetch), the startup `gc.collect` + `gc.freeze`, and the journal's threaded retag, parse cache, single regime query and debounced filters | Contained in packet 3 and therefore in `main`. Deleted local + origin |
| `claude/desk-snappiness-1` | 4 | `3ba49ea` | Snappiness packet 1: the health-audit evidence cache, the bounded column fit, and the Auto Pilot status memo | Contained in packet 2 and therefore in `main`. Deleted local + origin |
| `claude/theta-premium` | 5 | `19a4a7a` | Phase 0.11 theta premium optimization, T1-T7: the credit floor as a percent of the strike, support-first ranking with an uncapped spread penalty, 15 market days for credit spreads, a premium-capacity-ordered quote budget, the `premium=` report line and its Qt columns, and the spread credit scaling with the underlying | **Merged into `main` 2026-08-31** (merge commit `fad97d6`). Contained; deleted local + origin |

Every one of the four passed `git merge-base --is-ancestor <branch> main` before
deletion. `claude/gui-phase-0-9` was not part of that integration - but see the
correction below: it is CONTAINED in `main` and what is open is its gate, not the
branch.

## The 2026-09-02 integration (review round R1)

Eleven branches, all cut from `66a0c31`, all now on `main`. Phase 0.12, P3 and P7
merged in the morning; R1 fixed the blockers on the rest and merged the remaining
eight the same day.

| Branch | Tip | What it held | Disposition |
|---|---|---|---|
| `claude/focus-declutter-lrsi-htf` | `ce275b1` | Phase 0.12 A+B: the Focus surfaces stop growing, and the higher-timeframe LRSI entry study (16 shadow recipes) | merged into `main` 2026-09-02; containment proven |
| `claude/p0-apply-decisions` | `e4dc2fe` | P0: BANGER retired, LRSI M5 alerts silenced with every row of evidence kept, click-away recorded as a pass | merged into `main` 2026-09-02; containment proven |
| `claude/p1-grade-what-you-said` | `9439385` | P1: the swing-like cohort, the capture-time like merge, versioned veto pooling, and 640 decisions the scoreboard was reading as silence | merged into `main` 2026-09-02; containment proven |
| `claude/p2-show-me` | `774296c` | P2: the Decisions pane, the five AI phase gates, the M5 take rate and the repetition fold | merged into `main` 2026-09-02; containment proven |
| `claude/p3-fact-pack-truth` | `eb7bd43` | P3: the nightly fact pack states its own evidence shape (BD-81..85 after the merge renumber) | merged into `main` 2026-09-02; containment proven |
| `claude/p4-swing-variables` | `56b17d2` | P4: twelve tracker attributes, the finer leaderboard views, the shipped tier - plus R1's three blocker fixes | merged into `main` 2026-09-02; containment proven |
| `claude/p5-pass-cohorts` | `5d0a9f9` | P5: the pass and rejection cohorts - plus R1's cohort rename, pooled-row label and the missing merge test | merged into `main` 2026-09-02; containment proven |
| `claude/p6-preference-to-trade` | `abaf966` | P6: exact-id auto-tag candidates, the preference/trade report, the honest empty dimension - plus R1's four blocker fixes | merged into `main` 2026-09-02; containment proven |
| `claude/p6a-tag-backlog` | `b303423` | P6a: provisional tags, the bulk tagger and the review surface (24 tags applied to the live journal) | merged into `main` 2026-09-02; containment proven |
| `claude/p7-setup-registry` | `cb271b0` | P7: the setup crosswalk and the trial ledger - plus R1's registered_at | merged into `main` 2026-09-02; containment proven |
| `claude/p8-param-grid` | `9629f78` | P8: the first setup-parameter grid - plus R1's ledger wiring, the real assertion and the memoisation | merged into `main` 2026-09-02; containment proven |

**Every one passes `git merge-base --is-ancestor <branch> main`** - checked on
2026-09-02, all eleven. They are LEFT IN PLACE rather than deleted: nothing asked
for the cleanup, and the proof above is what makes deleting them safe whenever
somebody does.

**Corrected 2026-09-02 (R2): `claude/gui-phase-0-9` is CONTAINED in `main` too.**
`git merge-base --is-ancestor claude/gui-phase-0-9 main` succeeds and it is 0
commits ahead - its content reached `main` through `claude/group-tape-rebuild`,
which was cut from it. Two documents called it unmerged for weeks. What is open
is **gate 7 (SOAK 1)**, which is owed by work that HAS landed; a gate and a
branch are different things, and conflating them left a branch on the deletion
list's wrong side.

## The 2026-09-05 cleanup (trader: *"clean up the repo as needed"*)

The rule at the bottom of this file was finally applied. Every branch below passed
`git merge-base --is-ancestor <branch> main` against `main` at `ee325704` on
2026-09-05 ~21:30 PT, and was then deleted on `origin` and locally; 42 registered
worktrees whose HEAD was likewise contained (agent worktrees under `.claude/worktrees/`
and other sessions' scratch worktrees under the Temp scratchpads) were removed with
`git worktree remove --force`, and their throwaway `worktree-agent-*` branches (100 local
refs, every one contained) went with them. What each branch built is in `CHANGELOG.md`;
the dated checkpoint entries and `docs/archive/` still name them, on purpose.

**Deleted on `origin` and locally (42), by family, with the tip each held:**

| Family | Branches (tip) | Landed in |
|---|---|---|
| Phase 0.13 P-packets | `claude/p0-apply-decisions` (`e4dc2fe`), `p1-grade-what-you-said` (`9439385`), `p2-show-me` (`774296c`), `p3-fact-pack-truth` (`eb7bd43`), `p4-swing-variables` (`56b17d2`), `p5-pass-cohorts` (`5d0a9f9`), `p6-preference-to-trade` (`abaf966`), `p6a-tag-backlog` (`b303423`), `p7-setup-registry` (`cb271b0`), `p8-param-grid` (`9629f78`), `p9-quick-like` (`1f3f32b`), `p10-after-the-like` (`d8e310f`) | the 2026-09-02 integration (tabled above) |
| Review rounds and vision | `claude/r2-guards` (`739ea96`), `r3-narration-budget` (`6deac75`), `r4-fixes` (`0fad834`), `vision-2026-09-02` (`0b073db`), `v1-names-first` (`d5f012f`), `v2-loop-closes` (`35f64b1`), `v3-keep-it-honest` (`2cc94d4`) | 2026-09-02/03 |
| Desk and capture | `claude/strength-board-into-desk` (`bded98d`), `swing-favorites`, `daytrade-pass-reasons` (`f36ab59`), `focus-declutter-lrsi-htf` (`ce275b1`), `focus-refresh-storm` (`d38ec01`), `f1-desk-freeze` (`a736c1c`), `t1-capture-and-board` (`10db042`), `t2-claim-double-click` (`83f8df3`) | 2026-08-31 to 2026-09-04 |
| Phase 0.18 Q-packets | `claude/q1-held-honesty` (`f01601e`), `q2-warehouse-eligibility` (`a801396`), `q3-ai-grounding` (`8c5900f`), `q4-overnight-gates` (`2270769`), `q5-scorecard-worker` (`4a88df9`) | `b0db9bbe`, 2026-09-04 |
| Phase 0.19 / 0.21 M-packets | `claude/m1-band-variant-handoff` (`e744afd`), `m2-unresolved-means-unmeasured` (`de34346`), `m4-lake-band-variant` (`325ad7e`), `m5-shown-and-read` (`05cad91`) | `918445f3`, 2026-09-05 evening (the other lead session) |
| Phase 0.20 N-packets | `claude/n1-sidecar-aware-read` (`c09d944`), `n2-synthesis-output-cap` (`db5a8f0`), `n3-research-narration-bounded` (`a9ec8ff`) | `b907a801`, 2026-09-05 19:47 PT |
| Process | `claude/agent-team` (`e4dc48a`), `agent-team-2` (`3b5ea44`), `last-commit-main-dpouod` | 2026-09-02/03 |

**Local-only branches deleted (contained):** `claude/gui-p1-fluidity` (`88a34b7`),
`gui-phase-0-9` (`48c0ad4`), `group-tape-rebuild` (`cd212bc`), `local-ai-context-64k`
(`75880d6`), `warehouse-build-memory` (`ae8282d`), `phase05-integration-blitz` (`1a2fbde`),
`phase05-r2-focus-gating-strength-board` (`a8c696a`), `testing-week-2026-08-24` (`ed277a7`),
`lead/merge-n` (`b907a80`, the N merge worktree's branch), plus the 100 `worktree-agent-*` refs.

**Kept, and why:**

| Branch | Tip | Why it stays |
|---|---|---|
| `claude/s1-quick-verbs` | `0d51053` | NOT contained: S1.1/S1.2 superseded by T1, S1.4 rebuilt as `f903ca4`, S1.3 (ONE Strength surface) still owed a decision - a fresh packet, never a merge (checkpoint). Its two agent worktrees and one detached worktree at the same tip stay with it. |
| `claude/m3-tracker-keeps-up` | `3881bcc` | CONTAINED (landed in `918445f3`), but its worktree is LOCKED by the other lead session; the refs go when that lock clears. |
| `claude/avwap-band-challenger` | `9fe444d` | local only, NOT contained; the 2026-08-26 challenger branch this file's first section already records as "not lost" - unchanged. |
| `testing-week-2026-08-17` | `170172b` | local only, NOT contained; the pre-consolidation chain's first link, kept as the record above says. |

One scratch worktree at a detached `b4ca820b` (another session's Temp scratchpad, not
contained) was left for that session.

## Rule going forward

A branch may be deleted once `git merge-base --is-ancestor <branch> main` succeeds —
that is the proof that deletion discards nothing. A branch that fails that check is
either merged first or recorded in the table above with the reason it stayed open.
