# 0004 — PySide6/Qt consumer UI, Tk retained as legacy during migration

Date: backfilled 2026-08-01

## Context
The original GUI is Tkinter. The product direction (docs/SHIP_READINESS.md,
docs/archive/GUI_PRODUCT_PLAN.md) is a consumer-grade desktop app with a themed shell, eventually
a `TradingBotV3.exe`.

## Decision
The new Trading Desk UI (`scripts/ui/`) is PySide6 + qtawesome + pyqtgraph and is
the default launch target. `scripts/gui.py --ui tk` keeps the legacy Tk app working
during migration. `PyQt5` stays installed only for the legacy `TickerMover.py`.

## Rationale
The migration itself is evident (consumer polish beyond what Tk offers, per
SHIP_READINESS). The specific choice of PySide6 over PyQt (or web/other toolkits)
is not documented — RATIONALE UNKNOWN - confirm with Aaron.

## Amendment (2026-09-03)

The Tk app, its shims (`scripts/gui.py`, `gui_app/`, `market_prep_gui/`, `market_prep_tab.py`, `journal_tab.py`, `master_avwap_lib/gui.py`, `bounce_bot_lib/gui.py`, `bounce_bot_lib/alerts.py`) and `TickerMover.py` were removed by the trader-authorized assessment packet F2. `PyQt5` left the dependency set with them; the frozen spec keeps its `PyQt5` exclude as a guard. The scanner CLI (`master_avwap.py --once` / `--loop`) moved into `master_avwap_lib/runner.py`. PySide6 is the only UI.
