# 0001 — Decision-support only, no order execution

Date: backfilled 2026-08-01

## Context
A trading system can either suggest trades or place them. The boundary determines
broker integration scope, risk surface, and how "wrong" the system is allowed to be.

## Decision
TradingBotV3 does everything except execute orders (plan.md sec 1). Broker order
routing is explicitly outside the roadmap; broker data imports may remain read-only
inputs to the journal. Enforced as a non-negotiable invariant (plan.md sec 5:
"No broker execution is added under this roadmap").

## Rationale
RATIONALE UNKNOWN - confirm with Aaron. The boundary is stated emphatically and
repeatedly in plan.md but the underlying reason (risk? trust-building? regulatory?)
is never written down.
