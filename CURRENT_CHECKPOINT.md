# Current checkpoint

This file is the frequently refreshed active-work, branch, and verification stamp.

- Implemented inventory and revision history: [`CHANGELOG.md`](CHANGELOG.md)
- Remaining work and gates: [`plan.md`](plan.md)
- Supporting-document roles: [`docs/README.md`](docs/README.md)

## Active work — read this before choosing a task

There may be only one active build item unless `plan.md` explicitly identifies an
elapsed evidence lane that can run in parallel.

| Field | Current value |
|---|---|
| Roadmap phase | **P0 — validate and merge the testing-week branch** |
| Active packet | **DOC-1 — documentation consolidation, AI workflow, roadmap ordering, and wishlist** |
| Scope | Markdown, plus comment/docstring-only corrections for decision 0015 |
| State | Complete in the working tree; awaiting trader review/commit |
| Next action after this packet | **Commit the working tree** (control-set files `CHANGELOG.md`, `CURRENT_CHECKPOINT.md`, `WISHLIST.md` are still untracked), then **P0.2–P0.4** live gates. P0.1 re-baseline is done — see below. |
| Do not start yet | Phase 1 cleanup or any Phase 2+ feature/foundation item |

A newly arriving AI resumes the active packet if it is unfinished. If it is complete,
it performs the stated next action. It does not select a different roadmap item
without explicit trader direction.

## Branch

- Branch: **`testing-week-2026-08-10`**
- Current commit before this documentation pass: **`1f41af1`**
- Base: `main` at `7d85a27`
- State: **not merged to `main`; no PR recorded**
- Testing-week intent: Mon–Wed Auto/Away and baseline observation; Thu–Fri
  live-session validation; merge only after a `plan.md` Section 6 day passes.

## Last full Windows desk gate

Recorded at the 2026-08-09/10 desk re-baseline (`60119e8`):

| Check | Result |
|---|---|
| pytest | **2611 passed, 7 subtests passed**, exit 0 |
| JUnit | 2618 cases, 0 failures, 0 errors, 0 skipped |
| smoke | **7/7**, exit 0 |
| frozen self-test | **29/29**, exit 0 |
| Python | repo-local uv-managed **3.12** environment |

The frozen run found a real packaging-roster conflict: `ai_jobs` was deliberately
excluded from the bundle but required by self-test. The roster was corrected and a
permanent disjointness test added. The 29/29 figure is therefore the correct current
expectation, not the older 30/30 text in historical handoffs.

## Changes after that gate

The following commits landed after the recorded full gate and require coverage by the
next normal full run; none changes the frozen package inventory:

- `07395a0` — Chart Review Setups column defaults hidden and can be restored.
- `bfc8850` — a late-opened alert receives current bars.
- `4907b6f` — a published best-swing report can notify the phone.
- `1f41af1` — the swing push stays quiet when no readable setups exist.
- documentation consolidation: `CHANGELOG.md` for implemented history, `plan.md` for
  remaining work, `docs/README.md` for classification, and the renamed
  `CURRENT_CHECKPOINT.md` for active state;
- mandatory AI read/update workflow in `CLAUDE.md`/`AGENTS.md`, phase-gated roadmap
  ordering, and the new non-authoritative `WISHLIST.md`.

The documentation packet does not change the recorded automated baseline. Markdown
verification consists of link resolution, `git diff --check`, control-document
consistency, and confirmation that tracked edits remain Markdown-only.

## Re-baseline and desk configuration — 2026-08-10 (evening)

**P0.1 is satisfied for the four post-gate commits above.** Full Windows run on the
working tree:

| Check | Result |
|---|---|
| pytest | **2647 passed, 7 subtests passed**, exit 0 (109s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging-trigger change since 29/29 |

Re-run after the decision-0015 documentation/comment pass: **2647 passed**,
**smoke 7/7**, unchanged. That pass edited Markdown, docstrings, comments, and two
user-facing strings only; no behavior, path, or test changed.

**Current baseline after the local-AI evidence-budget packet:**

| Check | Result |
|---|---|
| pytest | **2659 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

Twelve new tests cover the budget resolver and its fallbacks, the cloud ceiling
staying untouched, the derivation itself (worst-case retry prompt must fit the
context left after generation), the truncation tripwire firing/staying silent, and
ledger usage recording.

Three desk misconfigurations were found and fixed by inspecting the first
testing-week session's artifacts. All three were machine-local settings lost when the
old desktop was retired; none was a code defect:

1. **Designated writer was unset** — `autopilot_today.txt` had not published since
   2026-07-30, so the whole 2026-08-10 session produced no phone digest and no swing
   push. Fixed with `writer_role.py --designate-self` (NucBox_K8_Plus). The desk was
   restarted at 19:37 local to pick it up (new pid 17984, heartbeat live), and
   `writer_role.py` now resolves `designated_writer / may publish True`, exit 0.
   **Not yet proven end to end:** `hourly_away_report_slot_due` returns nothing once
   the hour is past the session close, so no publish was due at restart time.
   `writer_health.json` consequently still carries its pre-fix 15:18 payload — that
   file is rewritten on a *publish attempt*, not at startup, so a stale copy here is
   expected and is **not** evidence the fix failed.
2. **`research_store_dir` was unset** — the warehouse was fully disabled and captured
   nothing. Now `\\MINI-PC\Trading Bot Data\research_lake`, layout created, and the
   restarted desk is the first process to run with it enabled. Capture is proven by
   the next scan writing under the lake, not by configuration alone.
3. **ntfy was already configured and works** — verified by test push (`ok: True`) at
   both `default` and `urgent` priority. Delivery to the iPhone banner/sound is an
   iOS-side setting and is **not yet confirmed by the trader**.

**AI jobs repaired 2026-08-10 (evening).** The task now exits 0 when run through the
scheduler. Details in `CHANGELOG.md`; the live proof is the 22:00 window tonight, and
`%LOCALAPPDATA%\TradingBotV3\logs\ai_jobs-<date>.log` will now carry any failure.
Two AI-layer caveats remain unproven and must be checked against tomorrow's ledger:

- ~~Context smaller than the evidence cap~~ — **closed the same evening.** Local
  calls now cap evidence at `ai_local_evidence_budget_chars` (22,000) and a
  truncation tripwire fails loudly if the server still sees less than was sent. The
  cloud ceiling is untouched.
- ~~The large tier cannot load~~ — **accepted and designed around.** The local large
  tier is retired (plan sec 2); policy drafts and retros belong to the frontier
  model. Revisit triggers recorded: Ollama Vulkan allocator work, ROCm on gfx1103,
  or more RAM.
- **Phase 2 design packet is PROPOSED, not approved.** `docs/LOCAL_AI_AUTOMATION_PLAN.md`
  sec 6.4a. Its six open questions need trader answers before any digest code is
  written — question 1 ("what counts as winning": R at scenario close, MFE/MAE, or
  both) is a trading judgement and is the one the whole fact pack hangs on.

### What the next session must confirm

Four fixes are configured and unit-verified but have **not** completed a live cycle.
None could be proven on the evening of 2026-08-10; all resolve by 09:00 on 08-11:

| Fix | Proof to look for | When |
|---|---|---|
| Designated writer | `autopilot_today.txt.meta.json` names `NucBox_K8_Plus` with a current `verified_at` — it still names the retired `DESKTOP-IABHR62` at 2026-07-30 | 07:00 publish |
| Swing phone push | an ntfy notification carrying numbered swings | 09:00 (push start hour) |
| Research warehouse | new files appearing under the lake root | first scan |
| AI jobs | `ai_jobs-20260811.log` records a completed `ai_summary` / `ticker_briefs` | 22:00-06:00 window |

If the 07:00 publish does not happen, read `writer_health.json` first: it will then be
fresh, and its `reason` names the exact gate that refused.

Still open on the desk, not blocking the week:

- `technical_integrity_events.jsonl` is ~247 MB and is never pruned (~10 MB/session).
- Off-site backup: cloud sync was the only off-site Class A copy (decision 0015).

## Immediate live gates

- **P0.1:** ~~run the complete Windows automated gate~~ — **done 2026-08-10**
  (2647 passed / smoke 7/7). Re-run before merge if further code lands.
- **P0.2–P0.4:** run the single-main session checklist, Away/ntfy validation, and
  observability rollover.
- **P0.5:** run the durability mid-session restart/backfill drill.
- **P0.6:** start Local-AI's five-session clock and the warehouse broker/live/pilot
  sequence.
- **P0.7:** merge only after the live-validation day and applicable rechecks pass.

Do not add historical detail here. When a change lands, update `CHANGELOG.md`; when a
gate remains, update `plan.md`.
