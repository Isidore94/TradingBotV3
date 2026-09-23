# TradingBotV3 — AI guide

A Windows PySide6 desk for one trader's day and swing trading: market prep, D1
anchored-VWAP swing scans, intraday M5 bounce alerts, Auto/Away scanning with a phone
report, a trade journal, and a night-time local-AI review. **It never places orders.**

## Read this, then stop reading

1. `STATUS.md` — what is live, in flight, next, and owed (~3 KB).
2. `TODO.md` — only if you are choosing work.
3. `docs/RULES.md` — only the section for the area you are changing. Each rule exists
   because of a real incident.
4. The code. **The code is the fact**; if a doc disagrees, fix the doc.

`notes/` (gitignored, local only) holds the full history: old checkpoint, plan,
changelog, specs, and the long story behind every rule. Open one file there only to
answer one specific question. `WISHLIST.md` is the trader's notepad — ideas, never
authorized work.

Before planning or editing, name the exact active work item (`STATUS.md`/`TODO.md` when
applicable), what exists and remains, governing rules, expected files/tests, and whether
ask-first applies. Only the trader may promote a WISHLIST idea into authorized work.

## Commands

- Test: `.venv\Scripts\python.exe -m pytest tests/ -q` (~10k tests). While building, run
  your area only (`pytest tests/test_<area>*.py -q`); run the full suite before every
  commit and merge. Check pytest's exit code, not a piped tail. macOS/Linux Qt:
  `QT_QPA_PLATFORM=offscreen`.
- Lint: `.venv\Scripts\python.exe -m ruff check .` must pass before every commit. Fix the
  code, not the config.
- Smoke: `.venv\Scripts\python.exe scripts/smoke_check.py` (7/7). Selftest: `launch_gui.py --selftest`.
- Run: `trading_desk.cmd` (= `launch_gui.py`). **The desk runs from SOURCE on `main`**, so a
  pushed commit goes live at the next restart. One desk per machine.
- Install: the venv has no pip — `uv pip install -r requirements-dev.txt -c constraints.txt --python .venv\Scripts\python.exe`.
- The nightly AI lock (22:00–06:00 ET) makes ~40 `run_slots` tests fail; run the full
  suite outside that window.

## Code map

| Want to change | Open |
|---|---|
| App shell, pages | `scripts/ui/app.py`, `scripts/ui/panels/` |
| Alert review / charts | `ui/panels/alert_center_panel.py`, `ui/widgets/` |
| D1 swing scan (detector, ask-first) | `scripts/master_avwap.py`, `master_avwap_lib/` (`runner.py`, `legacy.py`) |
| M5 bounce detector (ask-first) | `scripts/bounce_bot.py`, `bounce_bot_lib/`, `m5_signal_engines.py` |
| Auto modes, scheduling | `autopilot_core.py`, `ui/services/autopilot_service.py` |
| Focus lists, adoption gate | `focus_adoption_gate.py`, `FocusPickStore` in `ui/services/` |
| Journal (trades) | `journal_store.py`, `journal_*.py`, `ui/panels/journal/` |
| Day/Week Review, Mentor | `ui/panels/day_review_panel.py`, `ui/widgets/trade_mentor_card.py`, `day_report_card.py` |
| Night AI jobs | `scripts/ai_jobs/` (`runner.py` owns slot order) |
| Setup Tracker / scoring | `master_avwap_lib/`, `setup_scoreboard.py`, `working_lately.py` |
| Research warehouse (shadow only) | `scripts/research_warehouse/` |
| Paths to live stores | `scripts/project_paths.py` (always use its constants) |
| Pre-market prep | `market_prep/` |
| Indicators (pure) | `scripts/indicators/` |

## Hard invariants — never break

- Decision support only: never add order execution.
- No detector/scoring/alert behaviour change without golden-result fixtures first. The
  legacy SPY pause and D1 wick alerts are the champions; shadow engines never influence
  live output.
- Never change `calc_anchored_vwap_bands`' σ formula.
- Completed bars only for state changes. Missing data is "unknown", never "confirmed".
- Never auto-remove a watchlist name the trader typed (the one exception: the
  after-close wipe of `longs.txt`/`shorts.txt`, decision 0020).
- `review_policy.json` ranks and annotates only — never add a suppression field.
- A failed evidence write loses the event, never the pick/trade. A **journal** write fails loudly.
- Nothing expensive on the Qt thread — including stylesheets.
- Point-in-time research uses only what was known then; timestamps carry timezones.
- One owner per timer/thread/job/shared export; a failed publish never destroys the last good report.

## Ask first (before any edit, even a comment)

`scripts/master_avwap_lib/legacy.py`, `scripts/bounce_bot_lib/*`,
`scripts/m5_signal_engines.py`, and any other file holding detector, scoring or alert
code. When unsure whether a file counts, ask.

## Safety

- The trader's desk runs from `C:\Users\Aaron\TradingBotV3` and other AI sessions share
  that checkout: check the branch before staging or pushing, never stash, build in a worktree.
- Live stores are read-only unless the task names a write: `C:\TradingBotData`,
  `%LOCALAPPDATA%\TradingBotV3`, `\\MINI-PC\Trading Bot Data`. Copy before experimenting.
- **Scratch scripts** (anything outside pytest importing `scripts/`): set BOTH
  `TRADINGBOTV3_DATA_DIR` and `LOCALAPPDATA` to scratch folders before the import, and
  abort if any `project_paths` root points at a live store. A scratch leaderboard export
  patches `MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE` itself, never an alias. (This
  has overwritten live files twice.)
- A live-store repair goes through a tested CLI, never an ad-hoc script.
- Never restart the desk or merge to `main` without the trader's word.
- If a scheduled task runs unmerged branch code on the desk, disarm it before switching branches.

## After a change

1. Commit message says what and why — the commit log is the real history. Branch per
   task, commit small and green, push after each commit.
2. One line in `CHANGELOG.md`.
3. Update `STATUS.md` only if what's live / in flight / next / owed changed. Keep it under 3 KB.
4. Changed a rule? Update its line in `docs/RULES.md`. A new live check the trader must
   do goes in `docs/GATES.md` as one line.
5. Update `docs/README.md` whenever a Markdown file is added, moved, removed or reclassified.

Never add a new status, plan, report or handoff `.md` — reports go in chat or an artifact.

## Frozen exe

The desk runs from source; a pushed commit is live at the trader's next restart, and the
exe is verification only. Do not rebuild per commit. Build and run the frozen selftest
before EVERY merge to `main` and whenever a listed packaging trigger is hit: a new
third-party dependency; non-`.py` runtime asset outside first-party trees plus `config/`;
new top-level `scripts/` package imported lazily; dynamic import by string in an uncollected
package; or change touching `__file__`, `ROOT_DIR` or `sys.path`. One pre-merge build can
satisfy both checks.
Build: `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm`, then
`dist\TradingBotV3\TradingBotV3.exe --selftest` must run and match the source selftest count.
Guard packaging with `tests/test_packaging_spec_drift.py`; keep its deliberate-omission
list disjoint from `selftest.LAZY_ENGINE_MODULES`. If the trader returns to the frozen exe,
a fix is not delivered until rebuilt. Read Smart App Control at
`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` ->
`VerifiedAndReputablePolicyState`; never rely on memory. Read `packaging/README.md` first.
Ask before spending the trader's time on any Smart App Control click-through.
Merge to `main` only after a live-session validation day passes its checklist and with the
trader's word. Run that checklist on the first session and do not tune thresholds from one
session. Never restart the desk or merge to `main` without the trader's word.

## Agents — role defaults

- **Claude:** lead Opus 5.5. `recon` = Sonnet (read-only lookups). `builder` = Opus in a
  worktree (fail-first test, then the fix). `reviewer` = Opus, one round, blockers only;
  skip it for docs/UI-only changes. `tester` = Opus, only for detector/scoring work.
- **Codex:** `.codex/config.toml` sets Sol for new-thread lead, Luna max for the default
  manager, and Luna xhigh for named `recon`, `builder`, `tester` and `reviewer` workers.
  The manager delegates every assigned change, then integrates and validates. Keep scopes
  narrow and use a worktree. These are defaults; explicit session or lead overrides win.

## Talking to the trader

Chat as if to a five-year-old: very short, simple words, one idea per sentence. Say what
you did, what is broken, and what they need to do. Detail goes in commits, not chat.
