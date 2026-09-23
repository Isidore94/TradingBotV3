# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-22

- **Live on `main`:** everything through the 2026-09-22 integration (`cf9c6b2a`):
  overnight AI repair (AI-R1/R2/R3), setup-score repair (SP1/SP2), alert review
  follow-up (AR-1/2A/2B/3), recap learning (TJ-17A–D), TJ-9E exit notes and the
  per-trade Mentor Save.
- **Desk:** it runs from source on `main`. It was down for the 09-22 update and was
  not restarted. The next start is the trader's call.
- **Last full suite:** 10,428 passed, 14 skipped, 1 known flaky test (G7 research tab,
  green when run alone). Ruff clean, smoke 7/7, source and frozen selftest 97/97.
- **Repo slim-down merged** 2026-09-22 (`6ea50405`): docs and agent settings only. History is in the local `notes/`.
- **Waiting for the trader's word:** chart scroll-back zoom fix on
  `claude/chart-wheel-zoom-2026-09-21` (`11c3a339`). It is not on `main`. Merge it only
  while the desk is down.
- **Next action:** read the live gates on the next real session and the next
  local-model night (`docs/GATES.md`, newest first: #182–189).
- **Trader actions owed:** the options-journal repair (gate #162: desk down, market
  closed, daytime), the Saturday large-model probe, Mentor confirmations, and live
  click checks.
- **Reviews owed:** the per-trade Mentor Save (`182f3e08`) merged without a reviewer
  round.
