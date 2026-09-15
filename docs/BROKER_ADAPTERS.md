# Broker Adapter Roadmap

Document role: **deferred architecture reference**. The root `plan.md` owns timing;
this file does not authorize order execution. Current roadmap location: P7.3.

The app is IBKR-first today. Multi-broker support should be added by introducing
a broker/provider boundary, not by scattering `if broker == ...` checks through
the scanners and UI.

## Target Shape

The trading engines should depend on app-owned interfaces such as:

- Market data provider: daily bars, intraday bars, quotes, option chains.
- Execution/journal importer: fills, commissions, accounts, positions.
- Connection profile: broker name, host/auth/session settings, feature flags.

IBKR then becomes one adapter behind those interfaces. Future adapters could be
Schwab, Tradier, Alpaca, Polygon/data-only, or CSV/manual import, depending on
which read-only data and journal-import workflows are needed. Order routing remains
outside the product boundary.

## First Interface To Extract

Start with market data because it is already partially provider-aware:

- Daily bars currently prefer IBKR and fall back to Yahoo.
- Intraday bars also have IBKR/Yahoo behavior.
- The scan result tracks daily-bar source.

Create an internal boundary around:

- `fetch_daily_bars(symbol, lookback)`
- `fetch_intraday_bars(symbol, timeframe/lookback)`
- `get_option_chain(symbol)`
- `get_option_quotes/contracts(...)`

The engines should receive a provider object or provider registry rather than
calling IBKR/Yahoo helpers directly.

## Suggested Package Layout

Keep this as a future migration target, not an immediate rename:

```text
scripts/
  brokers/
    __init__.py
    base.py          # protocols/dataclasses owned by the app
    ibkr.py          # wraps current ibapi behavior
    yahoo.py         # data-only fallback adapter
    registry.py      # selected broker/data-provider wiring
```

Later, once the repo becomes an installable Python package, this can move under
`tradingbot/brokers/`.

## Migration Order

1. Document all broker/data calls and mark which are data-only vs trading/journal.
2. Extract read-only market data providers first.
3. Move option-chain/theta quote logic behind an options provider.
4. Move journal execution import behind an execution importer provider.
5. Update Settings UI to manage broker profiles.
6. Only then add new broker implementations.

## Guardrails

- Keep provider interfaces small and use app-level dataclasses, not broker API
  objects, across the boundary.
- Preserve Yahoo/data-only fallback behavior for scans that do not require live
  IBKR data.
- Never make UI widgets talk directly to broker SDKs.
- Keep tests around the old IBKR behavior before swapping in adapters.

## The scan manifest: which provider served each scan, and how fresh its bars were

Added by WISHLIST 10A (2026-09-12). Per-scan provenance used to live only in the
machine-local run manifest (`diagnostics/run_manifests/`, keyed by run id) and in the
per-row `daily_bar_source` column of the feature CSV. Neither answers the question a
provider swap actually raises: *for the report on screen right now, which source served
how many symbols, and how fresh were the bars it returned?*

`scripts/master_avwap_lib/scan_manifest.py` writes that answer to
`project_paths.MASTER_AVWAP_SCAN_MANIFEST_FILE`
(`data/runtime/master_avwap_scan_manifest.json`, shared home, temp + rename) at the end of
every scan, with one append-only copy per scan in
`MASTER_AVWAP_SCAN_MANIFEST_HISTORY_FILE`:

| field | meaning |
|---|---|
| `run_id` | the `ManifestRecorder` run id, so this and the diagnostics manifest join |
| `status` | `ok` / `partial` / `failed` - `partial` = returned having fetched fewer symbols than `universe_size` |
| `started_at` / `finished_at` | aware market-local ISO-8601; the offset is always present |
| `universe_size` / `symbols_fetched` | what the scan set out to evaluate, and what it reached |
| `daily_bar_source_counts` | `{source: symbol count}`, read through `legacy._get_daily_bar_source` - `yahoo`, `ibkr`, `cache` |
| `latest_input_bar_session` | the newest **completed** daily bar any symbol carried, or `null` |
| `preview_bar_used` | some symbol's newest row was dated a session that had not closed |
| `daily_bars_forming_dropped` / `_invalid_dropped` | WS-FC1's write-guard counters for this scan |
| `outputs` | `[{name, path, rows}]` for `priority_setups`, `theta_puts` (put rows + PCS rows) and `d1_watchlist` |
| `error` | the failure path only |

Two things this is for when adapters land:

- **A source pin is visible.** The desk pins daily bars to Yahoo
  (`local_settings.json` `daily_bars_source: "yahoo"`), and `daily_bar_source_counts` is
  where that pin shows up as a fact rather than as an inference from a log. A new adapter
  that silently serves half the universe is one `read_manifest()` away from being seen.
- **Freshness is separated from provenance.** A provider that answers fast with stale bars
  and one that answers slowly with current ones look identical in a latency counter.
  `latest_input_bar_session` is asked of the exchange calendar (WS-FC1's
  `daily_bar_cache.last_completed_session`, 16:00 ET inclusive, no early-close model), so a
  forming bar is labelled `preview_bar_used` and never counted as a completed session.

A failed scan writes the manifest with `status: failed` and **touches no output file**, so
the last good report keeps its bytes and its mtime; the desk labels it stale rather than
republishing it. `record_scan` never raises - an evidence store may not cost the thing it
records.

Readers: `scan_manifest.freshness_line` (one sentence, used by the Setups status row and by
the System Health check `master_scan_freshness`) and
`python -m master_avwap_lib.scan_replay --symbol X --session YYYY-MM-DD` (read-only, run
from `scripts/`). The rules behind both: `docs/DESK_INTERNALS.md` "10A - last scan, latest
bar and shown report are three clocks".
