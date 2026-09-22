# Decisions

One line each. The full records are in `notes/docs/decisions/` (local only).

- 0001: Decision support only. No order execution, ever.
- 0002: Legacy detectors are the champions. New engines run in shadow and log JSONL until they pass the promotion ladder.
- 0003: IBKR is the primary market data source, with yfinance as fallback. The bar source is recorded per scan.
- 0004: PySide6/Qt is the only UI. (Tk and PyQt5 have since been removed.)
- 0005: Plain-file storage (txt, JSON, CSV, append-only JSONL). The cloud-sync part is superseded by 0015.
- 0006: Shared exports use writer leases with fencing generations (`writer_lease.py`).
- 0007: Only completed bars drive state transitions. A forming bar is a preview.
- 0008: The `calc_anchored_vwap_bands` σ formula is frozen.
- 0009: Golden-result fixtures come before any detector or scoring change.
- 0010: The AI review policy only annotates and ranks. `review_policy.json` has no suppression field.
- 0011: AI summaries are one-way, evidence-grounded, schema-validated and provider-neutral.
- 0012: Requirements are layered (core ⊂ gui ⊂ dev) and pinned by `constraints.txt`.
- 0013: The doc hierarchy is superseded by the 2026-09-22 slim-down. Now: the code, then AGENTS.md, then STATUS/TODO.
- 0014: The DAS research lake is an append-only store of immutable Parquet at `research_store_dir`.
- 0015: No cloud sync. `C:\TradingBotData` is local, and the DAS `\\MINI-PC\Trading Bot Data` is the durable tier.
- 0016: The trader's priorities, in order: (1) trade only from this program; (2) one-click likes teach the bot; (3) the bot works out why; (4) day trading is the biggest prize; (5) sharper swing setups; (6) two-tier AI; (7) it works on away days; (8) honest numbers, shadow first; (9) the desk stays fast. Swing trades are judged on win rate, day trades on MFE after a held level. This is the tie-breaker for every priority call.
- 0017: The tracker is mirrored to SQLite, shadow first. Readers move one at a time after gate #57.
- 0018: The overnight run goes deterministic, then digest, then narration, then model-gated slots.
- 0019: Tracker replay defaults are `gap_aware_v2`, `prior_session_v2` and `first_actionable_v2`. The old ones stay available by name.
- 0020: `longs.txt`/`shorts.txt` are wiped whole after each session's close.
- 0021: Day Review replaces the Market Journal and Daily Recap pages, Week Review comes first in Weekend Prep, and the night narrates.
