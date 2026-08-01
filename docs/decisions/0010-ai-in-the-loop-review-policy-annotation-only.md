# 0010 — AI-in-the-loop review policy; annotation-only, no suppression

Date: backfilled 2026-08-01

## Context
The Alert Center learns the trader's preferences from logged decisions
(`alert_review_events.jsonl` → `review_learning.py` scoreboard). Phase 2 needed a
mechanism to act on that evidence.

## Decision
Phase 2 is deliberately an AI review step, not hard-coded logic: an AI agent reads
the scoreboard and writes `review_policy.json` (rank deltas, annotations, watch
presets). The policy format has no suppression field by design; queue ordering is
gated to annotation-only (FIFO) until the Phase 3 identity/parity gate passes.
Full mechanics: `docs/REVIEW_LEARNING_LOOP.md`.

## Rationale
Evident in the operating guide: the user wants an AI to periodically judge what to
prioritize; house rule "mute -> CAUTION, focus picks always surface" forbids
auto-suppression; and episode folding by `(trade_date, symbol)` currently collapses
distinct theses into one sample, so ordering on that evidence would be premature.
