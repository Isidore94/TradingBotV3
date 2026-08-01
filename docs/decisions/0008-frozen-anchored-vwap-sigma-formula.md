# 0008 — `calc_anchored_vwap_bands` σ formula is frozen (running-deviation variant)

Date: backfilled 2026-08-01

## Context
Anchored-VWAP bands are the backbone of the Master AVWAP engine; band-relative
levels feed setup detection, scoring, alerts, and chart watches across the app.

## Decision
The σ (band width) calculation stays on the current running-deviation variant.
Swapping it — even for a "more correct" formula — is a non-negotiable invariant
violation (plan.md sec 5, CLAUDE.md).

## Rationale
The freeze is evident: every band consumer is calibrated to this variant, so a
formula change silently shifts all thresholds at once. Why the running-deviation
variant was chosen originally (vs. e.g. a session-stddev band) is not documented —
RATIONALE UNKNOWN - confirm with Aaron.
