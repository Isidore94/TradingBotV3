# Sol adversarial reproduction pass — 2026-08-24 build slate

Frozen 2026-08-25 on branch `testing-week-2026-08-24`; attacked range
`97b6ae7..9edbf83`, with repairs through `21fd55e`. Live stores were read only,
the journal database was opened only with SQLite `mode=ro`, and no IB method was
called.

## Verdict

Seven blocker classes were reproduced: the AWAY recap has no alert input, two outcome-recovery paths can lose truth, the restarted desk cannot reach the sweep's own due time, teardown could hide a failed stop, blank unresolved outcomes were called usable, and Questrade activities sent an invalid request shape. Three surgical non-fenced blockers were repaired in one commit each; the AWAY repair needs a session-aware ordered feed contract, while both outcome defects live in fenced `bounce_bot_lib/legacy.py`, so those four were reported rather than built. T4, T5, T6 and T8 survived the attacks, but no live gate is marked met.

## Blockers

| Target | Claim attacked | Exact reproduction | Class | Fixed or reported |
|---|---|---|---|---|
| T1 | An AWAY day ends in a populated recap and records no less than DESK outside the review queue. | C1; live comparison C2 | **PROVEN** | **REPORTED.** `MainWindow` never calls `AwayRecapPanel.set_alerts`; the separate ordinary/D1 lists also lack one ordered, session-scoped export contract. |
| T2 | `_signal_bar_dict` recovers the signal bar belonging to the event. | C3 | **PROVEN** | **REPORTED.** A shifted cache with duplicate closes returns the 06:30 bar for a 06:35 event. `scripts/bounce_bot_lib/legacy.py` is fenced and no ask-first authorization was supplied. The flat alert row/tier path itself survived (C9). |
| T3 | Milestone recovery cannot erase a fact already recorded for a trade. | C4 | **PROVEN** | **REPORTED.** The first row at the furthest milestone wins; a later/furthest `stop_hit=False` erased an earlier `stop_hit=True`. The live Monday sweep had zero same-rank disagreements among its 673 recovered events, so this is a reachable integrity defect, not a claim that Monday was misgraded. Fenced file. |
| T3 | With autorun ON, the restarted desk performs a second after-close sweep. | C5 | **PROVEN** | **REPORTED.** The sweep becomes due at close+35 (13:35), while the owning strategy loop is automatically paused at close+30 (13:30). At 14:21 the 2026-08-25 CSV held 656 registrations and zero finals, and coverage was still the 2026-08-24 row. Fenced file; the live mechanics canary failed and stays owed. |
| T7 | Hermetic teardown cannot hide a `BounceBot.stop()` failure. | C6 | **PROVEN** | **FIXED** by `0c62b63`. A stop exception is returned to the autouse fixture even when the thread joined first. |
| T9 | A blank unresolved final is not usable evidence. | C7 | **PROVEN** | **FIXED** by `8474383`. Blank/non-finite `close_r` is unsettled alongside the legacy zero/entry sentinel; it is not relabelled `fabricated_zero_v1`. The live 2026-08-24 read moved from 93 purported usable entry claims to 0 usable / 493 unsettled at the time of verification. |
| T10 | Questrade `/activities` receives valid DateTime bounds. | C8 | **PROVEN** | **FIXED** by `21fd55e`. The importer now sends inclusive offset-bearing Pacific bounds; the focused journal set passed 81 tests. R7 I2 still treats an unavailable completeness cross-check as failed coverage rather than claiming COVERED. |

## Reproduction command ledger

### C1 — production-window AWAY recap input (exit 1)

```powershell
& '.\.venv\Scripts\python.exe' -c "import os,sys,tempfile;root=tempfile.mkdtemp(prefix='sol-away-recap-');os.environ.update({'QT_QPA_PLATFORM':'offscreen','LOCALAPPDATA':root,'TRADINGBOTV3_DATA_DIR':root,'TRADINGBOT_DIAGNOSTICS_DIR':root,'TRADINGBOT_DISABLE_BACKGROUND_MAINTENANCE':'1'});sys.path.insert(0,'scripts');from PySide6.QtWidgets import QApplication;from ui.app import MainWindow,PAGE_SPECS;from ui.state import UiState;from ui.models.bounce import BounceAlert;app=QApplication.instance() or QApplication([]);w=MainWindow(UiState(workspace_mode='workspace'));p=w.trading_panel.alert_center;p._auto_mode_now=lambda:'AWAY';p.mover_state=lambda *a:'open';p.add_alert(BounceAlert(time_text='10:00:00',symbol='AAA',side='LONG',trigger='[S-TIER] VWAP reclaim',timeframe='5m',raw_text='[S-TIER] AAA: VWAP reclaim'));i=[x.title for x in PAGE_SPECS].index('AWAY Recap');w._select_page(i);print('backing',len(p._alerts),'recap_input',len(w.away_recap_panel._alerts));assert w.away_recap_panel._alerts,'AWAY Recap was never handed the Alert Center backing list'"
```

Output: `backing 1 recap_input 0`, then `AssertionError`.

### C2 — live AWAY versus the last explicit DESK day

```powershell
& '.\.venv\Scripts\python.exe' -c 'import sys,collections;sys.path.insert(0,"scripts");from review_events import load_review_events;rows=load_review_events();days=collections.defaultdict(list);[(days[str(r.get("trade_date") or "")].append(r)) for r in rows];
for d in ("2026-08-21","2026-08-25"):
 c=collections.Counter(str(r.get("action") or "") for r in days[d]);print(d,len(days[d]),dict(c))'
Select-String -Path "$env:LOCALAPPDATA\TradingBotV3\logs\autopilot.log" -Pattern '2026-08-21.*Auto profile -> DESK|2026-08-25.*Hourly Away'
```

The explicit DESK day (2026-08-21) had 485 review rows / 137 `shown`; the
2026-08-25 AWAY day had 40 rows / 0 `shown`, while still carrying 35
`focus_d1_flag`, 3 `level_fired`, and 2 `d1_event_fired` rows. The zero shown
impressions are correct for `P(take|shown)`—no chart was shown—but they cannot
rescue the missing recap input proved by C1.

### C3 — shifted duplicate-close cache (exit 1)

```powershell
& '.\.venv\Scripts\python.exe' -c "import sys; from datetime import datetime; from types import SimpleNamespace; sys.path.insert(0,'scripts'); from bounce_bot_lib.legacy import BounceBot; b=BounceBot.__new__(BounceBot); b.get_cached_5m_bars=lambda s:[SimpleNamespace(dt=datetime(2026,8,25,6,30),open=98,high=103,low=97,close=100),SimpleNamespace(dt=datetime(2026,8,25,6,35),open=99,high=101,low=98,close=100)]; fallback={'time':'20260825  06:35:00','open':100,'high':100,'low':100,'close':100}; got=b._signal_bar_dict('AAA',0,fallback); print(got); assert got is fallback, 'shifted cache with duplicate close recovered the wrong 06:30 bar for a 06:35 event'"
```

### C4 — conflicting milestone rows erase a stop (exit 1)

```powershell
& '.\.venv\Scripts\python.exe' -c "import sys,tempfile;from pathlib import Path;sys.path[:0]=['scripts','tests'];import test_outcome_sweep as t;p=Path(tempfile.mkdtemp(prefix='sol-milestone-disagree-'));t._csv(p,[{'event_id':'a','event_type':'3_bar','close_r':'-1.0','mae_r':'-1.2','stop_hit':'True','bars_elapsed':'3','logged_at':'2026-08-21T08:00:00'},{'event_id':'a','event_type':'12_bar','close_r':'0.5','mae_r':'-0.2','stop_hit':'False','bars_elapsed':'12','logged_at':'2026-08-21T09:00:00'}]);h=t._host(p);t._seed(h,{'a':t._state(event_id='a')});c=h.sweep_pending_bounce_outcomes(now=t.AFTER_CLOSE);r=h.rows[-1];print(c);print({'status':r['status'],'stop_hit':r['stop_hit'],'close_r':r['close_r'],'context_json':r['context_json']});assert r['stop_hit'] is True,'later milestone erased an earlier recorded stop hit'"
```

Output classified the row `last_measured_bar:legacy_csv_milestones` with
`stop_hit=False`, then failed the assertion.

### C5 — second production sweep missed its own clock

```powershell
Get-Item "$env:LOCALAPPDATA\TradingBotV3\diagnostics\outcome_sweep_coverage.json" | Select-Object LastWriteTime,Length
Get-Content "$env:LOCALAPPDATA\TradingBotV3\diagnostics\outcome_sweep_coverage.json" -Raw
Select-String -Path "$env:LOCALAPPDATA\TradingBotV3\logs\trading_bot.log" -Pattern '2026-08-25 13:30.*Scanning paused|2026-08-25.*Outcome sweep'
& '.\.venv\Scripts\python.exe' -c 'import pandas as pd,sys;sys.path.insert(0,"scripts");from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE as p;cols=["event_id","event_type","trade_date"];parts=[]
for c in pd.read_csv(p,usecols=cols,chunksize=200000,low_memory=False):
 q=c[c.trade_date.astype(str)=="2026-08-25"]
 if len(q):parts.append(q)
f=pd.concat(parts,ignore_index=True);print("rows",len(f),"events",f.event_id.nunique(),"types",f.event_type.value_counts().to_dict())'
```

At 14:21 PT: coverage mtime `2026-08-24 16:47:18`, `swept_at`
`2026-08-24T16:45:43-07:00`; log `2026-08-25 13:30:11 Scanning paused` and
no 2026-08-25 sweep; CSV `656 registered`, `0 final`.

### C6 — teardown fail-before and repaired test

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests/test_suite_hermetic_teardown.py::test_a_stop_exception_is_reported_even_if_the_thread_joined
```

Fail-before: returned `[]` after logging the exception. Fixed: pass.

### C7 — blank unresolved final fail-before and live query

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests/test_setup_scoreboard.py::test_a_blank_unresolved_final_is_not_usable
& '.\.venv\Scripts\python.exe' -c "import sys;sys.path.insert(0,'scripts');import setup_scoreboard as s;from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE as p;f,c=s.load_intraday_finals(p,window_start='2026-08-24',window_end='2026-08-24');print(c)"
```

Fail-before: `coverage.unsettled == 0`, `usable == 1` for the synthetic row.
Fixed live read at execution time: `in_window=493`, `unsettled=493`, `usable=0`.

### C8 — Questrade request fail-before and focused repair gate

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests/test_journal_backfill.py::test_questrade_activities_use_full_offset_datetimes
& '.\.venv\Scripts\python.exe' -m pytest -q tests/test_journal_backfill.py tests/test_journal_cash_and_options.py tests/test_journal_coverage.py tests/test_journal_nightly_slot.py
```

Fail-before sent `2026-08-17` / `2026-08-24`; the official Questrade sample
uses full offset-bearing DateTimes. Fixed parameters are
`2026-08-17T00:00:00-07:00` / `2026-08-24T23:59:59-07:00`; focused result
`81 passed`, exit 0. Source: [Questrade activities API](https://www.questrade.com/api/documentation/rest-operations/account-calls/accounts-id-activities).

### C9 — unchanged tier and H1 basis boundary

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q tests/test_r5_lrsi_cross_wiring.py::test_the_outcome_registration_gets_the_real_signal_bar tests/test_r5_lrsi_cross_wiring.py::test_a_registerable_risk_now_exists_for_an_lrsi_cross tests/test_r5_lrsi_cross_wiring.py::test_the_alert_row_and_the_tier_still_see_the_flat_bar tests/test_h1_entry_time_is_the_bar_close.py
```

Result: `10 passed`, exit 0.

## REFUTED-by-attack

**REFUTED-by-attack:** T2's alert row/tier and H1 basis boundaries (C9); T3's cross-process/write-ahead/idempotence mechanics (`8 passed` focused and Monday's 687 finalized / 0 failed / 0 commit-failed); T4's deleted-role callers and surviving Focus privilege (`56 passed`); T5's real 16,384-byte fail-closed cap (24,574-byte pack, zero files) and advisory import isolation (`3 passed`); T6's per-session FX/missing-rate/idempotence contract (`8 passed`) and golden diff (only the three additive USD keys plus its note); T8's n/robust-half/claim-language sample (zero nesting violations, no `realizable` claim); T10's I2 downgrade semantics; and IBKR Flex as a request defect (the same request succeeded at 22:00 with 369 executions before the later server-side “try again shortly” failures).

## Appendix — non-blocking evidence

- C5 also preserves Monday's result: 687 pending, 687 finalized, 7 expired,
  0 failed, 0 commit-failed, 1 already recorded, 673 recovered, 0 pending.
  Its 686 newly appended final rows have one final per event, 395 stop hits with
  `stop_exit_r`, 7 `no_measurement_in_checkpoint` rows unresolved and numeric
  close on none. The extra recorded-existing row reconciles 686 CSV appends to
  the coverage count of 687.
- Questrade activities failure correctly makes completeness unknown under R7
  I2; the ledger has no `UNKNOWN` state, so `FAILED` is the honest status. The
  request-shape defect, not that semantic, was repaired.
- Two full post-fix suites produced identical pass sets: `4801 passed, 19
  subtests`, exit 0, then the same again; JUnit comparison printed
  `only_run1 0 only_run2 0`. Smoke was 7/7, source selftest 70/70, and the
  unchanged frozen executable selftest 70/70, all exit 0.
- No packaging trigger fired: no module/package/asset was added to the frozen
  bundle, and the desk runs from source. The frozen selftest was verification,
  not a rebuild.
