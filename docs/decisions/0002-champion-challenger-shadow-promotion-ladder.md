# 0002 — Champion/challenger architecture with a shadow-evidence promotion ladder

Date: backfilled 2026-08-01

## Context
New engines (SPY pullback `market_state`, `greatness_monitor`) could replace legacy
detectors that the trader already relies on live. Swapping silently risks regressing
live behavior with no evidence trail.

## Decision
Legacy detectors stay the "champions". New engines run in shadow via bridge modules
(`market_state_bridge`, `greatness_shadow`) that log JSONL evidence but cannot affect
live output. Promotion requires the plan.md sec 7 ladder: versioned config,
deterministic tests, replayable evidence, live-session evidence across regimes,
champion comparison, and a rollback switch that needs no code revert.

## Rationale
Evident in plan.md sec 2: "No feature may silently promote itself from research or
shadow mode into production decision-making", and "agreement with the legacy
implementation is diagnostic, not the definition of correctness."
