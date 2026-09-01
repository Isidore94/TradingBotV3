# Branch history and the consolidation to `main`

Last reconciled: **2026-08-31**

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
production through a scheduled task (`docs/CHECKPOINT_REVIEW_2026-08-08.md`). Each
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
| `claude/trade-analysis-opus-prompt-vgg1n8` | 1 | 2026-08-22 | `6c1398f` | One additive document, merged into `main` on 2026-08-26 as `docs/prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md` |

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
   `docs/ALERT_CENTER_QUALITY_PACKET.md` — a live spec for the packet it builds —
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
deletion. `claude/gui-phase-0-9` was deliberately left alone: it is separate
long-running work with its own open gate (SOAK 1) and was not part of this
integration.

## Rule going forward

A branch may be deleted once `git merge-base --is-ancestor <branch> main` succeeds —
that is the proof that deletion discards nothing. A branch that fails that check is
either merged first or recorded in the table above with the reason it stayed open.
