# 0007 — Completed bars only for state transitions

Date: backfilled 2026-08-01

## Context
Intraday engines (M5 bounce detection, SPY pullback state machine, chart watches)
consume live bars that are still forming and can repaint before completion.

## Decision
Only completed bars may satisfy completed-bar confirmation rules or drive state
transitions; a forming bar is preview only. Shadow evidence accounting counts an
evaluation as usable only when driven by a truly completed bar (see
`market_state_bridge.py`); forming-bar and data-gap cases get their own counters.

## Rationale
Evident in plan.md sec 5 (Data and time): forming bars repaint, so acting on them
produces signals that later un-happen; paired with "missing data is uncertainty,
never silent confirmation" and the point-in-time research rule to keep live behavior
and research free of look-ahead.
