# 0011 — One-way, provider-neutral, evidence-grounded AI advisory summaries

Date: backfilled 2026-08-01

## Context
The app has an A.I. workspace that summarizes bot/journal artifacts. An LLM with
write access could corrupt scanner state or smuggle unvalidated claims into live
surfaces.

## Decision
`scripts/ai_summary.py` (and `market_prep/services/ai_service.py`) package selected
artifacts as evidence, request schema-constrained JSON from a provider (OpenAI
Responses API or Anthropic Messages structured output), validate every evidence
reference locally, and export the result. No function in the module can write
scanner state, scores, watchlists, alerts, or orders.

## Rationale
Fully documented in the `ai_summary.py` docstring: "deliberately one-way" with
local validation of every evidence reference; provider contracts verified
2026-07-14. Consistent with invariants 0001 and 0002.
