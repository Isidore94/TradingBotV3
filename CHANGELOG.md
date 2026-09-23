# Changelog

One line per merged change, newest first. The commit message holds the detail. Keep
about 40 lines and delete the oldest. The full old changelog and inventory are in
`notes/CHANGELOG.md`, local only.

- 2026-09-22 Setup grades: PROVEN/A/B/C/D/New from the trackers' real results (swing: tracker win rate + low bound; day trade: +1R before -1R). Shown on the setups table and M5 bar, best first; the priority switch now defaults ON.
- 2026-09-22 Repo slim-down: AGENTS.md is 8 KB, CLAUDE.md imports it, and the
  checkpoint/plan became STATUS/TODO. History moved to the gitignored `notes/`. Agents
  are cheaper: Opus/Sonnet on Claude, gpt-6-luna/gpt-6-sol on Codex.
- 2026-09-22 Four work streams integrated on main (`cf9c6b2a`): AI-R1/R2/R3 overnight
  AI repair, SP1/SP2 setup-score repair, AR-1/2A/2B/3 alert review follow-up,
  TJ-17A–D recap learning.
- 2026-09-21 Per-trade Mentor Save: an answered trade is stored at once and never asked
  again (`182f3e08`).
- 2026-09-21 TJ-9E: the Mentor tells an exit from an entry. One exit box, plus a night
  slot that drafts why/felt/watching blind to the outcome (`7230b30d`).
- 2026-09-20 TJ-7: mood and process fields, reported only, never acted on (`b63db7af`).
- 2026-09-20 TJ-6: up to three grounded AI ideas a night, kept only by the trader's
  click (`a7809d7c`).
- 2026-09-20 TJ-5: Week Review first in Weekend Prep (`1b9d77e0`).
- 2026-09-20 TJ-12: a six-line report card heads Day Review (`843a3f02`).
- 2026-09-20 TJ-4: the day pack and the overnight day story (`d929e34f`).
- 2026-09-20 TJ-16: prediction ledger vs baselines, plus a blind word tagger (`c4a760e5`).
- 2026-09-20 TJ-14B: Mentor question registry, budget of three (`161e905c`).
- 2026-09-20 TJ-10: read grader, prediction ledger, congruence lines (`57b44ca9`).
- 2026-09-20 TJ-9Q: Questrade fill classifier; a sold put is a sale (`86c64f96`).
- 2026-09-20 TJ-13B: night-only large-model probe (`e54c8203`).
- 2026-09-19 TJ-14A: Mentor card with a forced prediction click (`e8c04f88`).
- 2026-09-19 TJ-15: D1 miss contrast (`fb3f55e9`).
- 2026-09-19 TJ-3: note markers on Day Review charts (`72647104`).
- 2026-09-19 TJ-11/11F: walk-away v2 and the decision session (`a89ec7d5`, `f00ec302`).
- 2026-09-19 TJ-9: forced 09:00 trade labels (`8077a758`).
- 2026-09-19 TJ-13A: night-only local inference, night slates (`9eaae1dd`).
- 2026-09-18 TJ-2A/2B: session bars and walk-away tables.
- 2026-09-18 TJ-1/1L: one Day Review page in two columns (`e00b734a`).
- 2026-09-16 Phase 0.32 entry quality finished (`5bc09528`).
