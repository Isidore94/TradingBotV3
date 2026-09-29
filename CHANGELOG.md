# Changelog

- 2026-09-29 Movers: the Dip boxes' anchors are flipped back (trader): Dip-strong measures from SPY's low since its last big M5 drop (else the low of day), Dip-weak from its high since its last big M5 rip (else the high of day). An anchor under 30 minutes old keeps last tick's ("held"), else the low/high of day if old enough, else the open. Gate #303 replaced by #309.

- 2026-09-29 Desk: the Qt thread no longer waits on the BounceBot child for status reads (IB connected, auto regime, entry assist, market environment, pacing). One proxy poller refreshes them every 2 s and the GUI reads the cache; before the first answer, or after 30 s without one, they read as unknown. Today's log had 40+ s of 1-2 s GUI stalls from `refresh_health` / `refresh_auto_regime` / Auto status queued behind the RPC lock at bar close.

- 2026-09-29 Desk: the strength board build now runs in its own below-normal child process, not a desk thread (its 30-36 CPU-s/min starved the GUI of the GIL: 23.7 s, 16 s, 13.6 s freezes). Same cadence, board and status; a failed, crashed or 10-minute-timed-out build keeps the last good board, and closing the desk kills a running build. `launch_gui` now calls `multiprocessing.freeze_support()` so frozen spawn children work.

- 2026-09-29 Night AI: one pass and one recheck, then done (trader). A clean scheduled pass ends the night; otherwise the next firing is the only recheck. Then the model is unloaded from the 5080, a host the night woke is shut down at once (05:30 stays a backstop), and later firings exit without waking anything (`night_passes-<night>.txt`, `night_done-<night>.flag`). The task now last fires at 05:30, not 06:00 (re-run `register_ai_jobs_task.ps1`). The 07:00 retry and `--status` never touch the host. Every host ssh call has a timeout and closed stdin: a hung check held the 06:00 run open for 30+ minutes on 2026-09-29.

- 2026-09-28 Desk: removed the "Best right now" box from the M5 column and its "Show: Best right now" filter choice (trader: it listed weak names); a saved Best choice falls back to Grade B and up. Gates #231/#232 retired.
