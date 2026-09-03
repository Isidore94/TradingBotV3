# The agent team: one lead, its builders, its reviewers

Document role: **active runbook**. How a Claude Code session in this repo plans, builds,
reviews and integrates work using project-defined sub-agents instead of pasting prompts
between windows. The agent definitions live in `.claude/agents/` (tracked); this file is
the contract they share with the lead session and with the trader.

## The roles

| Agent | Model | Where it runs | What it may do | What it must never do |
|---|---|---|---|---|
| **lead** (the session the trader talks to; Fable by default) | session model | the main checkout | ask the trader, write packets, spawn the others, merge, run the full suite, reconcile the ledgers | build a packet itself when a builder could, restart the desk without the trader's word |
| **builder** (`.claude/agents/builder.md`) | Opus, effort high | its own worktree, branch `claude/<slug>` off `main` | edit, test, commit, push its branch, reconcile docs on its branch | touch the main checkout, merge, delete branches, edit a detector/scoring file without the trader's recorded yes |
| **tester** (`.claude/agents/tester.md`) | Opus, effort high | its own worktree, on the packet's branch | write the packet's tests, prove each FAILS on the current code, commit them red | write the fix; weaken or skip a test |
| **reviewer** (`.claude/agents/reviewer.md`) | Opus, effort high | its own worktree, checked out to the branch under review | run tests, revert-and-rerun to prove fail-before-fix, reproduce claims on COPIES of live data, render Qt offscreen | write, edit, commit, push, touch live stores |
| **recon** (`.claude/agents/recon.md`) | Sonnet, effort medium | the main checkout, read-only | map code with file:line, count live rows, find gaps | write anything, propose designs unasked |

The built-in `Explore` and `Plan` agents remain available for one-off lookups; `recon`
is the same job with this repo's rules baked in.

## The loop

1. **Recon first.** Before the lead writes a packet it spawns `recon` on the premises
   ("does X exist, where, what does the live file show"). A packet is written only from
   verified premises - the 2026-08-24 lesson is that two of that review's claims were
   refuted at code level by the builder.
2. **The packet.** The lead writes it as a numbered list: the trader's decision quoted,
   the facts with file:line, the exact change per item, the tests that must fail first,
   the invariants that bind, the docs to reconcile, the live gate. Packets P0-P10 and
   V1-V3 (2026-09-02) are the models; the house-rules block is now inside
   `builder.md`, so a packet no longer repeats it.
3. **Tests first, for anything that matters.** For a packet with more than one item, or
   any item touching evidence, scoring, alerts, the journal or a trader-facing screen,
   the lead spawns `tester` FIRST. It writes one test per item that drives the real
   path, proves each fails on the current code, and commits them red. The builder then
   makes them pass and may only ADD tests. Round 3 of 2026-09-02 found four tests that
   could not fail, all written by the agent that wrote the fix; this is the cure.
4. **Build.** The lead spawns `builder` with the packet and the branch slug, in the
   background. One builder per packet. Two packets that touch the same files run one
   after the other, not in parallel - the 2026-09-02 integration of seven parallel
   branches cost an evening of conflict resolution.
   **The lead checks the handoff against the diff before believing it.** `git diff
   --stat <base>..<branch>` and the item list must agree: an item marked "done" with no
   file behind it, or a file changed that no item names, is a question for the builder
   before any reviewer is spawned. Two handoffs on 2026-09-02 said "built" for items
   that had no code.
5. **Review by reproduction.** The lead spawns `reviewer` with the branch, the packet
   and the builder's handoff. GO / NO-GO with blockers separated from advisories.
   Review rounds 1 and 2 of 2026-09-02 each found real defects the green suite had
   passed (a NaN read as a tier named "NAN"; a link tag evicting real setup tags), so
   this step is never skipped for a packet that touches evidence or the desk.
6. **Fix round.** Blockers go back to a builder as a small fix packet on the same
   branch (`SendMessage` to the same builder keeps its context; a fresh builder gets the
   reviewer's blockers verbatim). Advisories are batched into a later packet.
7. **Integrate.** The lead merges to `main` in a SCRATCH WORKTREE, never in the desk's
   checkout while the desk is up, then runs the full suite with the nightly AI lock
   FREE (probe it; 32 tests stand down while it is held), ruff, smoke and the source
   selftest, and refreshes the checkpoint block. Merge order is the packet order.
8. **The desk.** A merged commit reaches the desk only at its next restart, and the
   restart is the trader's call. The lead says in one line that it is owed and why.

## Rules that exist because something broke

- **One checkout, many agents.** The desk runs from `c:\Users\Aaron\TradingBotV3`.
  Builders and reviewers work in worktrees under `.claude/worktrees/`. Nobody switches
  the main checkout's branch while the desk is running: on 2026-09-02 the desk died
  during the after-close wrap-up with the working tree mid-merge under it.
- **The nightly AI lock.** From ~22:00 until the run finishes (it took six hours on
  2026-09-01) `test_ai_jobs_runner.py`, `test_ai_evidence_coverage.py` and
  `test_ai_jobs_store_window.py` stand down. A suite run with the lock held is not a
  baseline. Probe: `local_writer_lock('ai_jobs_runner', timeout_seconds=0.0)`.
- **Fail-before-fix is proven, not claimed.** The builder restores the pre-change file
  and watches the new test fail; the reviewer does it again independently.
- **Old rows have the key PRESENT and EMPTY, not absent.** A test that models an old
  row with a missing key does not model the real file (the "NAN" tier, R2).
- **A fixture is pinned from the OLD code.** A fixture generated by the code it is
  meant to pin is a self-portrait (P8's fixture builder pins commit `1837b63`).
- **Live stores are read-only to every agent** except a builder whose packet names the
  write. The one authorised exception so far was P6a's provisional tagging, and it
  took a backup first (`trade_journal.sqlite3.p6a-backup-*`).
- **Detector, scoring and alert files are ask-first.** The trader's decision quoted in
  the packet for the exact functions is the answer; otherwise the builder stops and
  the question goes in the handoff. `CLAUDE.md` lists the files.
- **Chat to the trader is short.** Detail lives in commits, docs and handoffs.

## Delegation policy for the lead

The lead's job is routing, not typing. The cheapest correct agent does each job.

- **Do it yourself:** reading; a lookup under a minute; `git status/log/diff`; committing
  and pushing work that already exists on a branch; merging in a scratch worktree;
  doc-only edits under ~40 lines; answering the trader.
- **Spawn `recon` (Sonnet):** any question that needs more than three files read, or a
  count from a live store. Never Opus for a lookup.
- **Spawn `tester` then `builder`:** any packet with more than one item, or any item
  that touches evidence, scoring, alerts, the journal or a trader-facing screen.
- **Spawn `builder` alone:** a one-item packet the lead can verify by running one test.
  Small packet (one file, under ~80 lines): pass `model: sonnet` at call time. Otherwise
  Opus.
- **Spawn `reviewer`:** every builder branch that touches evidence, scoring, alerts, the
  journal or a trader-facing screen. Skip for docs-only branches and for one-line fixes
  the lead verified by running the test.
- **Packets live in `.claude/packets/<name>.md`.** The lead hands an agent the file
  path, never the pasted text, so the lead's own context stays small.
- **Between jobs the trader runs `/clear`.** The checkpoint block is the memory, not the
  chat.

## How the trader uses it

- "Recon: <question>" - the lead spawns `recon` and reports the answer.
- "Build packet <name>" - the lead writes or reuses the packet, spawns `tester` then
  `builder` per the policy, checks the handoff against the diff, and reports.
- "Review <branch>" - the lead spawns `reviewer` and reports GO / NO-GO.
- "Integrate" - the lead merges in order, runs the gates, and says whether a restart
  is owed.
- The trader can also `@builder` or `@reviewer` a task directly in the chat.

Costs: recon is Sonnet and cheap; builders and reviewers are Opus at high effort and
each packet-sized run is a real spend. The lead does not spawn a reviewer for a docs-only
branch, and does not spawn two builders on the same files.

## Setup on a machine

1. The agent files are tracked under `.claude/agents/` (`.gitignore` un-ignores that
   folder; the rest of `.claude/` stays machine-local). A fresh checkout has them.
2. `.claude/settings.json` (machine-local, not tracked) allow-lists the commands the
   agents run without a prompt: pytest, smoke, selftest, ruff, `git worktree`,
   `git checkout -b`, `git commit`, `git push` to `origin claude/*`. The snippet is
   in the section below; anything not listed prompts the trader, which is intended
   for destructive commands.
3. No plan, flag or restart is needed: a new or changed file in `.claude/agents/` is
   picked up by the running session.

### settings.json allow rules for the team

```json
"Bash(C:/Users/Aaron/TradingBotV3/.venv/Scripts/python.exe -m ruff check *)",
"Bash(.venv/Scripts/python.exe -m ruff check *)",
"Bash(C:/Users/Aaron/TradingBotV3/.venv/Scripts/python.exe launch_gui.py --selftest)",
"Bash(git worktree *)",
"Bash(git checkout -b claude/*)",
"Bash(git checkout claude/*)",
"Bash(git add *)",
"Bash(git commit *)",
"Bash(git push -u origin claude/*)",
"Bash(git push origin claude/*)",
"Bash(git diff *)",
"Bash(git log *)",
"Bash(git status *)",
"Bash(git show *)"
```
