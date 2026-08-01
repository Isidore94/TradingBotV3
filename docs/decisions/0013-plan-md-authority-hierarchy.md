# 0013 — Documentation authority hierarchy: plan.md > GUI plan > checkpoint stamps

Date: backfilled 2026-08-01

## Context
Multiple agents (Codex "Sol" line, Claude) work this repo in sequence; conflicting
or duplicated status docs previously drifted (a 2026-07 commit reconciled plan.md
status claims against repository evidence).

## Decision
`plan.md` is the single source of truth for roadmap, status vocabulary, invariants,
and promotion policy. `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` is a subordinate
addendum that never overrides plan.md secs 5-7 or the sec 12 order.
`SOL_PROGRESS.md` / `GUI_LEARNING_PROGRESS.md` are small checkpoint stamps that
must not duplicate their plans.

## Rationale
Evident in the files themselves ("This file is only the... checkpoint stamp; it
must not duplicate the roadmap", "Authority order is unchanged") and in commit
545f475 reconciling status claims with verified evidence: duplication drifts,
so status lives in exactly one place per program.
