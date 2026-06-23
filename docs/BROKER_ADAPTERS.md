# Broker Adapter Roadmap

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
which workflows need live data, option chains, executions, or order routing.

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
