# Changelog

- 2026-09-29 Night AI: one pass and one recheck, then done (trader). A clean scheduled pass ends the night; otherwise the next firing is the only recheck. Then the model is unloaded from the 5080, a host the night woke is shut down at once (05:30 stays a backstop), and later firings exit without waking anything (`night_passes-<night>.txt`, `night_done-<night>.flag`). The task now last fires at 05:30, not 06:00 (re-run `register_ai_jobs_task.ps1`). The 07:00 retry and `--status` never touch the host. Every host ssh call has a timeout and closed stdin: a hung check held the 06:00 run open for 30+ minutes on 2026-09-29.

- 2026-09-28 Desk: removed the "Best right now" box from the M5 column and its "Show: Best right now" filter choice (trader: it listed weak names); a saved Best choice falls back to Grade B and up. Gates #231/#232 retired.
