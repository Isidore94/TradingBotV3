# 0009 — Golden-result fixtures required before any detector/scoring change

Date: backfilled 2026-08-01

## Context
The legacy scanner cores (`master_avwap_lib/legacy.py`, `bounce_bot_lib/legacy.py`)
are large, live-relied-upon, and being refactored incrementally. Ordinary unit tests
cannot prove that a refactor preserved exact scan output.

## Decision
No detector or scoring behavior change lands without golden-result
(characterization) fixtures first: recorded inputs with pinned expected outputs,
plus a replay harness (plan.md Milestone 3, testing strategy sec 10).

## Rationale
Evident in plan.md: the champions' current behavior is the spec, so fixtures make
any behavior drift a loud test failure instead of a silent live regression. The
fixture contract (immutable, versioned artifacts) also serves the promotion ladder's
replayable-evidence requirement (decision 0002).
