# Regime Infrastructure Phase 1 — first live-session validation

Status: **IMPLEMENTED, NOT LIVE-VALIDATED — EXPLORATORY / NON-PROMOTABLE**

This build starts data collection. It does not alter Technical Integrity scores,
alerts, rankings, watchlists, setup behavior, AVWAP math, or any execution path.
No signal may be promoted from this evidence until at least 40 instrumented
sessions exist (60 preferred) and a point-in-time predictive study beats the
declared trivial baselines.

## Before the session

1. Restart the desktop GUI on `regime-infrastructure-phase1`.
2. Confirm the application log contains a breadth-contract verification line.
   On the primary account as probed 2026-07-30, IBKR did not expose VOLD. It
   qualified `TICK-NYSE@NYSE`, conId `26718738`, as the first breadth proxy with
   usable historical M5 bars. The downgrade is logged at CRITICAL and every row
   says `proxy_kind=nyse_tick_proxy`; it is never mislabeled as true VOLD.
3. Keep the mini-PC Auto Pilot off, per the existing single-desktop IB budget.
4. Do not tune any threshold from this session.

## During the session

- At 10:30 ET (one hour after the NYSE open), confirm the technical ledger gets:
  - `frozen_intraday_snapshot`
  - `opening_range_baseline`
- At 12:00 ET, confirm a second `frozen_intraday_snapshot`.
- If the process was not alive at either target, the correct result is
  `missed_snapshot`, not a reconstructed snapshot.
- The machine-local `vold_m5.jsonl` should gain one `breadth_bar` per completed
  M5 bar, plus explicit `data_gap` rows where applicable.
- Do not restart merely to fix a missing row during the session; preserve the
  failure evidence.

## After the close

Run:

```powershell
.venv\Scripts\python.exe scripts\regime_collection_audit.py
```

For full evidence:

```powershell
.venv\Scripts\python.exe scripts\regime_collection_audit.py --json
```

The audit must show:

- all new resolutions have `post_resolution_tracking_started`;
- each completed chain has +30/+60/+90 follow-up events, or an explicit data
  gap;
- near-close windows are present with `truncated: true` and actual bar counts;
- both frozen targets have either a live snapshot or an honest missed marker;
- the opening-range baseline exists;
- breadth bars have unique bar-end timestamps and actual contract provenance;
- every collection event has `code_version`, `as_of`, and `written_at`.

An `UNHEALTHY` result blocks the live-validation exit gate. A `HEALTHY` result
validates collection mechanics only; it does not make any signal promotable.
