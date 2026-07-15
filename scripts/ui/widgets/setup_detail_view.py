from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTextBrowser

from setup_docs import build_trade_plan, resolve_setup_doc
from ui import theme
from ui.panels.setup_docs_panel import render_doc_html
from ui.services.ai_state_levels import load_symbol_levels
from ui.widgets.research_explanation_view import render_research_explanation_html


class SetupDetailView(QTextBrowser):
    """Right-hand setup detail: family mechanics + this symbol's stop/TP prices.

    Shared by the Trading Desk setups table and the Research Setup Tracker.
    Anchor-band levels come from the last scan's ai_state, loaded lazily on a
    background thread and cached; the pane re-renders itself when they land.
    ``playbook_lookup(side, family)`` optionally supplies the family's
    measured-best stop/exit variant to append to the plan.
    """

    _levelsLoaded = Signal()

    def __init__(self, parent=None, *, playbook_lookup: Callable[[str, str], dict | None] | None = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setMinimumWidth(340)
        self.setVisible(False)
        self._playbook_lookup = playbook_lookup
        self._symbol_levels: dict[str, dict] = {}
        self._levels_loading = False
        self._current: dict[str, Any] | None = None
        self._levelsLoaded.connect(self._on_levels_loaded)

    def set_playbook_lookup(self, lookup: Callable[[str, str], dict | None] | None) -> None:
        self._playbook_lookup = lookup

    # ------------------------------------------------------------------
    def show_setup(
        self,
        *,
        symbol: str = "",
        side: str = "LONG",
        setup_family: str = "",
        favorite_signals: list | tuple = (),
        tier: str = "",
        last_close=None,
    ) -> None:
        self._current = {
            "symbol": str(symbol or "").strip().upper(),
            "side": str(side or "LONG").strip().upper(),
            "setup_family": setup_family,
            "favorite_signals": list(favorite_signals or []),
            "tier": str(tier or "").strip().upper(),
            "last_close": last_close,
            "research_kind": "",
            "research_row": None,
        }
        needs_levels = bool(self._current["symbol"]) and not self._symbol_levels and not self._levels_loading
        if needs_levels:
            self._levels_loading = True  # before render so the first paint says "loading"
        self._render()
        if needs_levels:
            threading.Thread(target=self._load_levels_worker, daemon=True).start()

    def show_family(self, setup_family: str, side: str = "") -> None:
        self.show_setup(symbol="", side=side or "LONG", setup_family=setup_family)

    def show_research_row(self, kind: str, row: dict[str, Any]) -> None:
        self._current = {
            "symbol": "",
            "side": str(row.get("side") or row.get("direction") or "LONG").strip().upper(),
            "setup_family": str(row.get("setup_family") or ""),
            "favorite_signals": [],
            "tier": str(row.get("tier") or "").strip().upper(),
            "last_close": None,
            "research_kind": str(kind or ""),
            "research_row": dict(row),
        }
        self._render()

    # ------------------------------------------------------------------
    def _load_levels_worker(self) -> None:
        try:
            self._symbol_levels = load_symbol_levels()
        finally:
            self._levels_loading = False
            try:
                self._levelsLoaded.emit()
            except RuntimeError:
                pass  # the view was deleted while levels loaded in the background

    def _on_levels_loaded(self) -> None:
        if self._current is not None:
            self._render()

    def _render(self) -> None:
        current = self._current
        if current is None:
            return
        symbol = current["symbol"]
        side = current["side"]
        doc_key, doc = resolve_setup_doc(current["setup_family"])

        body_open = f"<body style='color:{theme.color('text_primary')}; font-size:9pt'>"
        parts = [body_open]
        if current.get("research_row") is not None:
            parts.append(
                render_research_explanation_html(
                    str(current.get("research_kind") or "research"),
                    current.get("research_row"),
                    include_body=False,
                )
            )
        if symbol:
            side_color = theme.color("long" if side == "LONG" else "short")
            tier_text = (
                f"<b style='color:{theme.color('favorite')}'>{_esc(current['tier'])}</b> "
                if current["tier"]
                else ""
            )
            parts.append(
                f"<h2 style='margin:0'>{tier_text}{_esc(symbol)} "
                f"<span style='color:{side_color}'>{_esc(side)}</span></h2>"
            )
            parts.append(self._plan_html(current))
        if symbol or current.get("setup_family"):
            doc_html = render_doc_html(doc_key, doc, heading_level=3)
            parts.append(doc_html.replace(body_open, "").replace("</body>", ""))
        parts.append("</body>")
        self.setHtml("".join(parts))
        self.setVisible(True)

    def _plan_html(self, current: dict[str, Any]) -> str:
        muted = theme.color("text_secondary")
        favorite = theme.color("favorite")
        short_c = theme.color("short")
        long_c = theme.color("long")
        symbol = current["symbol"]
        side = current["side"]
        family = current["setup_family"]

        levels = self._symbol_levels.get(symbol) if self._symbol_levels else None
        if levels is None:
            if self._levels_loading:
                return f"<div style='color:{muted}'>Loading level data from the last scan...</div>"
            return (
                f"<div style='color:{muted}'>No level data for {_esc(symbol)} in the last scan's state file — "
                f"family guidance below still applies.</div>"
            )

        plan = build_trade_plan(
            side=side,
            setup_family=family,
            favorite_signals=current["favorite_signals"],
            bands=levels.get("bands") or {},
            vwap=levels.get("vwap"),
            atr20=levels.get("atr20"),
            last_close=_float(current.get("last_close")) or levels.get("last_close"),
        )

        def _price(value) -> str:
            return f"{value:,.2f}" if isinstance(value, (int, float)) else "n/a"

        def _r(value) -> str:
            return f"{value:+.1f}R" if isinstance(value, (int, float)) else "n/a"

        parts = [f"<h3 style='margin:8px 0 2px 0; color:{favorite}'>How to execute it, step by step</h3>"]
        parts.append(
            f"<div><b>1. Entry reference:</b> {_price(plan.get('entry_reference'))} "
            f"<span style='color:{muted}'>(last scan close; anchor {_esc(levels.get('anchor_date'))})</span></div>"
        )
        parts.append(
            f"<div><b style='color:{short_c}'>2. Invalidation/stop:</b> {_esc(plan['stop_label'])} @ {_price(plan.get('stop_price'))} "
            f"<span style='color:{muted}'>— fires after {plan['stop_close_failures']} daily close(s) beyond it; "
            f"{_esc(plan['stop_reason'])}</span></div>"
        )
        risk = plan.get("risk_per_share")
        if isinstance(risk, (int, float)):
            pct = plan.get("risk_pct_of_price")
            pct_text = f" ({pct:.1f}% of price)" if isinstance(pct, (int, float)) else ""
            parts.append(f"<div><b>3. Risk per share (1R):</b> {_price(risk)}{pct_text}</div>")
        else:
            parts.append(
                f"<div style='color:{short_c}'>Price is already beyond the stop level — the plan is stale; "
                f"wait for the next valid trigger.</div>"
            )
        parts.append(
            f"<div><b style='color:{long_c}'>4. First target (take 50%):</b> {_esc(plan['partial_label'])} @ "
            f"{_price(plan.get('partial_price'))} <span style='color:{muted}'>({_r(plan.get('partial_r'))})</span></div>"
        )
        parts.append(
            f"<div><b style='color:{long_c}'>5. Runner target:</b> {_esc(plan['final_label'])} @ "
            f"{_price(plan.get('final_price'))} <span style='color:{muted}'>({_r(plan.get('final_r'))}); trail stop to "
            f"{_esc(plan['trail_label'])} after TP1</span></div>"
        )
        parts.append(f"<div style='color:{muted}'>Time stop: {plan['time_stop_sessions']} sessions.</div>")
        parts.append(
            f"<div style='color:{muted}; margin-top:4px'><b>Novice glossary:</b> 1R is the planned loss "
            "between entry and stop. A completed-bar close is required where the setup says so; an intraday "
            "wick or preview is not confirmation.</div>"
        )

        best = self._playbook_lookup(side, family) if self._playbook_lookup else None
        if best:
            parts.append(
                f"<div style='color:{muted}; margin-top:4px'><b>Measured best variant for this family:</b> "
                f"stop {_esc(best.get('stop_reference_label'))}, {_esc(best.get('profit_take_summary'))} "
                f"(robust {_signed(_float(best.get('robust_closed_r')) or 0.0)}R over "
                f"{_int(best.get('closed_setups'))} closed).</div>"
            )
        return "".join(parts)


def _esc(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _signed(value: float, *, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}"
