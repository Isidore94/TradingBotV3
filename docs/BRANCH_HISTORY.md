# Branch history and the consolidation to `main`

Last reconciled: **2026-08-26**

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
| `phase05-integration-blitz` | 308 | 2026-08-04 → 2026-08-23 | `1a2fbde` | Contained in `testing-week-2026-08-24`. Branch deleted 2026-08-26 |
| `testing-week-2026-08-17` | 262 | 2026-08-04 → 2026-08-20 | `170172b` | The previous week's release candidate. All but one commit contained in `testing-week-2026-08-24`; the exception is a doc-reconciliation note superseded by the 2026-08-25 reconciliation. Branch retained for now |
| `phase05-r2-focus-gating-strength-board` | 150 | 2026-08-04 → 2026-08-18 | `a8c696a` | R2 Focus gating and the M5 strength board. Contained in `testing-week-2026-08-24`. Branch deleted 2026-08-26 |
| `claude/ticker-briefs-hardening-imcm8r` | 94 | 2026-08-04 → 2026-08-11 | `9e0df9e` | Ticker-brief hardening and the first night's measurements. Contained in `testing-week-2026-08-24`. Branch deleted 2026-08-26 |
| `claude/trade-analysis-opus-prompt-vgg1n8` | 1 | 2026-08-22 | `6c1398f` | One additive document, merged into `main` on 2026-08-26 as `docs/prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md` |

The three branches marked deleted held **no commit** that is not on `main`. That was
verified with `git merge-base --is-ancestor` before deletion, not assumed.

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

## Rule going forward

A branch may be deleted once `git merge-base --is-ancestor <branch> main` succeeds —
that is the proof that deletion discards nothing. A branch that fails that check is
either merged first or recorded in the table above with the reason it stayed open.
