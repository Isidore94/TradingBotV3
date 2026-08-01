# 0003 — IBKR primary market data, Yahoo Finance fallback

Date: backfilled 2026-08-01

## Context
The scanners need daily and intraday bars for hundreds of symbols; a single feed
failing mid-session would blind the whole system.

## Decision
Market data comes from a local IBKR TWS/Gateway session (`ibapi`, `127.0.0.1:7496`)
first, falling back to `yfinance`. The scan result records which source supplied
each daily-bar set. A future provider boundary (`fetch_daily_bars` /
`fetch_intraday_bars`) is sketched in `docs/BROKER_ADAPTERS.md` so other brokers
become adapters, not scattered `if broker == ...` checks.

## Rationale
The fallback design is evident (resilience + source provenance). Why IBKR is the
primary feed is not written down — RATIONALE UNKNOWN - confirm with Aaron
(presumably it is the broker Aaron trades through, per README prerequisites).
