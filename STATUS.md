# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25

- **Live on `main`:** the whole 2026-09-24 wishlist plan (P0-1 to P2-11), in three merges on 2026-09-25: batch 1 (P0-2 rest, P1-4 setup keys, P1-5 best-right-now, P1-6 entry plan, P1-7 trading plan; #222-#241), batch 2 (P2-8 breadth, P2-9 looking back, P2-10 journal; #242-#250), batch 3 (M5 setup-key sidecar #252, P2-11 cleanup: swallowed-error logging, ruff E7/E9/F/B, Health universe + IB rows, secrets in Credential Manager, lazy legacy boot; #253-#256; bounce_bot `main()` sys fix). Reviewers GO. What is left is in `WISHLIST.md`.
- **Night AI fixes** merged 2026-09-25 (`af1cc04c`, frozen selftest 104/104): pytest -n capped to 2 22:00-02:00 PT after the 09-24 low-memory bluescreen; grounded ideas, story, enrichment tags and setup research. Gate #257. Owed: TLT/USO have no daily bars (scan-side, ask first).
- **Desk:** runs from source on `main`. Restart is the trader's call. First start after batch 3 moves the OpenAI key and ntfy token into Windows Credential Manager. New dependency `keyring` is installed in `.venv`.
- **Last full suite:** 2026-09-25 batch 3 integration: 11,872 passed; 2 load flakes (pass alone) and `test_st6 ... empty_snapshot` (fails on main too). Ruff clean, smoke 7/7, selftest 104/104 source and frozen.
- **Owed from 09-23:** halted-name refetch; movers summary into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask first).
- **Next action:** read gates #217-#256 on the next session, close scan and night (`docs/GATES.md`). Fix the `test_st6` empty-snapshot failure.
- **Trader actions owed:** set Risk per trade ($) in Settings > General; fill in `trading_plan.md`; stamp the sector/industry maps (`map_freshness.py --apply`); desk down + market closed: `journal_pnl_repair.py` (#205) and the options-journal repair (#162); Questrade gap days via `journal_questrade_gaps.py --statement`; the Saturday large-model probe; live click checks.
- **Reviews owed:** the per-trade Mentor Save (`182f3e08`).
