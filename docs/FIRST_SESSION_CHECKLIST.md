# First-session validation checklist

Document role: **active acceptance runbook**

Applies to: `plan.md` Section 6 and P0.2–P0.5

Topology: **single main desk**

Last reconciled: **2026-08-10**

This checklist turns an implemented/green build into real operational evidence. A
deterministic test cannot satisfy a live row. Record failures and UNKNOWN states
exactly as observed; do not tune thresholds during the first session.

## A. Before the session

Record:

| Fact | Value |
|---|---|
| Date/session | |
| Branch and commit | |
| Machine/Windows/Python | |
| TWS or Gateway version/mode | |
| Home folder | |
| Research store enabled/path | |
| Auto mode | |
| Operator | |

Then:

1. Confirm only the main desk is authorized to run. Keep Desk Link role on Main and
   do not start the retired mini-PC scanner.
2. Confirm writer authority:

   ```powershell
   .venv\Scripts\python.exe scripts\writer_role.py --designate-self
   ```

3. Confirm the launch task is armed as intended and that the Local-AI task state is
   deliberate.
4. Start TWS/Gateway and confirm API market data.
5. Run the automated gate and record pytest's own exit code:

   ```powershell
   .venv\Scripts\python.exe -m pytest tests\ -q
   .venv\Scripts\python.exe scripts\smoke_check.py
   ```

6. If a frozen rebuild trigger applies, build and run:

   ```powershell
   dist\TradingBotV3\TradingBotV3.exe --selftest
   ```

   Current expected result: `selftest OK: 29/29 checks passed (frozen)`.
7. Launch with `launch_gui.py`; verify one process only.
8. Capture the initial System Health page. UNKNOWN is acceptable before the first
   instrumented event but must not be silently rolled up as HEALTHY.

## B. During the live session

Check each row from real behavior:

| Area | Required observation | Result/evidence |
|---|---|---|
| Runtime | heartbeat advances; one generation owns timers/threads | |
| Master scan | one run ID/PID; phases and provider counters recorded | |
| BounceBot | completed M5 bars drive transitions; forming data is preview | |
| SPY shadow | rows/coverage advance without changing champion pause state | |
| Greatness shadow | transition evidence advances without changing D1 alerts | |
| Provider telemetry | attempts/success/failure/fallback denominators are present | |
| Review capture | a real decision writes a valid per-installation v2 event | |
| Chart Review | LIKE/veto/note is quick and writes no Focus/watch/alert privilege | |
| Charts | D1/M5 refresh, source/age is truthful, fallback is loud | |
| Paint lines | groups toggle; level IDs persist; click-to-arm uses PriceAlertService | |
| Price alert | actual/test push reaches ntfy; fire disarms only that side | |
| Auto/Away | current verified digest publishes with correct section order | |
| Warehouse | live tee/Health tiles advance if enabled; disabled path is a no-op | |
| GUI | no sustained event-loop stall or main-thread I/O regression | |

For the first session/config rollover on the build, also verify:

- the prior SPY/Greatness log rotates before the new scope writes;
- session summaries reconcile with raw rows;
- eligible/incomplete counters and retention are honest;
- corrupt or truncated evidence makes the audit non-promotable rather than invisible.

## C. Durability restart drill

Run only when it is safe to interrupt the desk:

1. Confirm a pending Technical Integrity follow-up or breadth window exists.
2. Stop/restart the GUI mid-session through the normal launcher path.
3. Confirm the repeating task/single-instance guard never creates a second desk.
4. Allow the recovery sweep to finish.
5. Run:

   ```powershell
   .venv\Scripts\python.exe scripts\regime_collection_audit.py
   ```

6. Require HEALTHY with a nonzero backfill count, explicit `capture_mode`, no duplicate
   live row, and no IB pacing conflict with the champion scan.

Tier-C evidence (frozen snapshots, never-started predictions, opening observations)
must remain missed rather than reconstructed.

## D. After the close

Run and preserve:

```powershell
.venv\Scripts\python.exe scripts\operations_audit.py
.venv\Scripts\python.exe scripts\review_capture_audit.py
.venv\Scripts\python.exe scripts\regime_collection_audit.py
```

Inspect:

- `%LOCALAPPDATA%\TradingBotV3\diagnostics\run_manifests\`;
- `heartbeat.json` and `job_ledger.jsonl`;
- `spy_state_shadow.jsonl` and `greatness_shadow.jsonl` plus summaries;
- verified Away report and metadata;
- review-event shard and annotation stream;
- warehouse coverage/gap/backup status when enabled;
- AI job ledger/morning artifact when its scheduled gate is active.

Close the GUI and verify bounded, clean shutdown with no orphan process. Relaunch once
if restart persistence is part of the packet.

## E. Acceptance record

| Gate | PASS / FAIL / UNKNOWN | Artifact or note |
|---|---|---|
| Full automated gate | | |
| One-process runtime | | |
| Provider telemetry | | |
| Shadow rollover/audit | | |
| Review/annotation capture | | |
| Chart freshness/paint lines | | |
| Price-alert delivery | | |
| Verified Away publication | | |
| Durability restart/backfill | | |
| Warehouse live path | | |
| Clean shutdown/restart | | |

Only the rows with PASS and preserved real-session evidence may be changed to
`LIVE_VALIDATED` in `CHANGELOG.md`. Any still-open row remains in `plan.md`.
