# 0004 — PySide6/Qt consumer UI, Tk retained as legacy during migration

Date: backfilled 2026-08-01

## Context
The original GUI is Tkinter. The product direction (docs/SHIP_READINESS.md,
GUI_PRODUCT_PLAN.md) is a consumer-grade desktop app with a themed shell, eventually
a `TradingBotV3.exe`.

## Decision
The new Trading Desk UI (`scripts/ui/`) is PySide6 + qtawesome + pyqtgraph and is
the default launch target. `scripts/gui.py --ui tk` keeps the legacy Tk app working
during migration. `PyQt5` stays installed only for the legacy `TickerMover.py`.

## Rationale
The migration itself is evident (consumer polish beyond what Tk offers, per
SHIP_READINESS). The specific choice of PySide6 over PyQt (or web/other toolkits)
is not documented — RATIONALE UNKNOWN - confirm with Aaron.
