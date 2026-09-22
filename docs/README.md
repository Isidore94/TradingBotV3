# docs/

- `RULES.md`: the desk's rules by area. Read only the section you're changing.
- `GATES.md`: live checks still owed, one line each.
- `DECISIONS.md`: the accepted decisions, one line each.
- `DESK_TESTING_PLAN.md`: the trader's live-test runbook. The desk shows it under
  Settings ▸ Testing Plan, and it ships in the exe.
- `SETUPS_MAJOR.md`, `SETUPS_TEST.md`: setup family names. `ai_jobs/enrichment.py`
  reads these at runtime.
- `MACOS_SETUP.md`: running the desk on a Mac.
- `EVENING_MODE_RUNBOOK.md`, `AWAY_SCANNER_RUNBOOK.md`, `FIRST_SESSION_CHECKLIST.md`: trader runbooks.
- `analysis/scripts/`: lake audit scripts. Tests read these.

Everything else (old specs, plans, the checkpoint and changelog archives, and the long
rule stories in `DESK_INTERNALS.md`) is in the gitignored `notes/` folder on the desk
machine.
