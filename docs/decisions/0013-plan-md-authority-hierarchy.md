# 0013 — Documentation authority hierarchy: root truth > references > checkpoints

Date: backfilled 2026-08-01

Amended: 2026-08-10 documentation consolidation

## Context
Multiple agents (Codex "Sol" line, Claude) work this repo in sequence; conflicting
or duplicated status docs previously drifted (a 2026-07 commit reconciled plan.md
status claims against repository evidence).

## Decision
`CHANGELOG.md` is the single source of truth for implemented inventory and revision
history. `plan.md` is the single source of truth for remaining work, status
vocabulary, invariants, execution order, and promotion policy. `docs/README.md`
classifies all supporting documents. Detailed GUI/warehouse/AI plans are subordinate
references and never override plan.md Sections 5–7 or the Section 12 order.
`CURRENT_CHECKPOINT.md` is a small active-work/branch/test stamp and must not duplicate either
root truth file. `WISHLIST.md` is non-authoritative and cannot initiate work; only a
trader-directed promotion into `plan.md` changes the build sequence.
`docs/archive/GUI_LEARNING_PROGRESS.md` is retained only as a historical pointer.

## Rationale
The former arrangement still duplicated implemented status across `plan.md`,
the former checkpoint ledger, and several product plans. The 2026-08-10 consolidation splits
past and future deliberately: completed facts live once in `CHANGELOG.md`; unfinished
gates live once in `plan.md`; transient test numbers live in `CURRENT_CHECKPOINT.md`.
