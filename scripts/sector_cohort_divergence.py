#!/usr/bin/env python3
"""`sector_cohort_divergence` - a SHADOW-only sector cohort watch (plan.md R9.5).

**Ladder position: SHADOW, and it stops there.** Nothing in this module reaches a
detector, score, ranking, routing, alert, watchlist, Focus, the review queue or
`review_policy.json`. Its only output is append-only JSONL under
`diagnostics/shadow_evidence/sector_cohort/`.

Why it exists. On 2026-08-21, 25 of 26 electric utilities closed below their
open - mean -2.78%, XLU itself -2.57% - while SPY closed -0.05%. AEP, at -4.13%,
was merely the second-worst member of that cohort, and no surface on the desk
ever named the sector. The archetype scan that found AEP is worthless without
this: strip the utilities out of that session and it lost money.

The rule, per the review's section 6e:

* ~20 sector/industry ETFs, on every **completed** M5 bar,
  ``spread = (ETF move from session open) - (SPY move from session open)``;
* fire when ``|spread| >= 0.75%`` has persisted across **>= 3 consecutive
  completed bars**. The persistence clause is not decoration: 31 of the 179
  fires measured over 23 sessions sat on the 09:30 bar and were gap artifacts;
* session only, re-derived and never carried;
* an unknown sector excludes the symbol - it never counts as a match.

Data comes from batched yfinance over the ETF set, the M5 Strength Board's
template, so this spends **zero IB pacing budget**.

Gates it satisfies on delivery: 1 (versioned config with a stable
``config_hash``), 3 (coverage accounting on every run, including quiet ones),
7 (a single defaults-dict switch, shipped off). Gates 2, 4, 5, 6 and 8 are
unmet and are not addressable by building it: the evidence window is >= 40
sessions spanning bullish, bearish and chop, declared before inspection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

CONFIG_VERSION = "sector_cohort_v1"
SHADOW_SCHEMA = "sector_cohort_shadow_v1"
# Sessions are measured independently and never carried across a date boundary.
SESSION_ONLY = True

# The single switch (gate 7). It ships OFF: a thing at SHADOW that starts itself
# is not at SHADOW.
SECTOR_COHORT_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "version": CONFIG_VERSION,
    "benchmark": "SPY",
    "threshold_pct": 0.75,
    "persistence_bars": 3,
    "min_bars_for_a_session": 12,
    # Member entry timing - the archetype, called rather than restated.
    "entry_window_et": ["10:00", "11:30"],
    "opening_high_bars": 3,
    "first_hour_bars": 12,
    "swing_stop_lookback_bars": 6,
    "etfs": [
        "XLU", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE",
        "XLC", "SMH", "IBB", "XBI", "KRE", "ITB", "XRT", "OIH", "XOP", "GDX",
    ],
}


def resolve_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Defaults with optional overrides. The result is what gets hashed."""
    config = json.loads(json.dumps(SECTOR_COHORT_DEFAULTS))
    for key, value in (overrides or {}).items():
        config[key] = value
    config["version"] = CONFIG_VERSION
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable identity for a configuration (gate 1).

    ``enabled`` is excluded: turning the watch on and off is an operational act
    and must not read as a different engine when the evidence is compared.
    """
    payload = {key: value for key, value in sorted(config.items()) if key != "enabled"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CohortObservation:
    session: str
    etf: str
    direction: str  # "short" when the ETF is weaker than the benchmark
    first_fire_bar_index: int
    first_fire_at: str
    spread_at_fire_pct: float
    max_abs_spread_pct: float
    final_spread_pct: float
    qualifying_bars: int

    def as_row(self) -> dict:
        return {
            "session": self.session,
            "etf": self.etf,
            "direction": self.direction,
            "first_fire_bar_index": self.first_fire_bar_index,
            "first_fire_at": self.first_fire_at,
            "spread_at_fire_pct": round(self.spread_at_fire_pct, 4),
            "max_abs_spread_pct": round(self.max_abs_spread_pct, 4),
            "final_spread_pct": round(self.final_spread_pct, 4),
            "qualifying_bars": self.qualifying_bars,
        }


@dataclass(frozen=True)
class MemberEntry:
    symbol: str
    entry_time_et: str
    fill: float
    stop: float
    risk_pct: float

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_time_et": self.entry_time_et,
            "fill": round(self.fill, 4),
            "stop": round(self.stop, 4),
            "risk_pct": round(self.risk_pct, 4),
        }


@dataclass
class DetectionResult:
    observations: list[CohortObservation] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# bar helpers
# ---------------------------------------------------------------------------
def _stamp(bar: Mapping[str, Any]) -> datetime | None:
    value = bar.get("dt")
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _session_of(bar: Mapping[str, Any]) -> str:
    stamp = _stamp(bar)
    return stamp.date().isoformat() if stamp is not None else ""


def _float(bar: Mapping[str, Any], key: str) -> float | None:
    try:
        return float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None


def _session_bars(bars: Sequence[Mapping[str, Any]], session: str) -> list[Mapping[str, Any]]:
    return [bar for bar in bars or [] if _session_of(bar) == session]


def _moves_from_open(bars: Sequence[Mapping[str, Any]]) -> list[float | None]:
    """Cumulative % move from the session open, per completed bar."""
    if not bars:
        return []
    opening = _float(bars[0], "open")
    if opening is None or opening <= 0:
        return [None] * len(bars)
    out: list[float | None] = []
    for bar in bars:
        close = _float(bar, "close")
        out.append(None if close is None else (close - opening) / opening * 100.0)
    return out


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def detect_cohorts(
    *,
    benchmark_bars: Sequence[Mapping[str, Any]],
    etf_bars: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> list[CohortObservation]:
    """Cohort observations for the benchmark's session. Never raises."""
    return run_detection(
        benchmark_bars=benchmark_bars, etf_bars=etf_bars, config=config
    ).observations


def run_detection(
    *,
    benchmark_bars: Sequence[Mapping[str, Any]],
    etf_bars: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> DetectionResult:
    """Detection plus the coverage accounting gate 3 asks for."""
    threshold = float(config.get("threshold_pct", 0.75))
    persistence = int(config.get("persistence_bars", 3))
    min_bars = int(config.get("min_bars_for_a_session", 12))

    coverage: dict[str, Any] = {
        "config_version": config.get("version"),
        "etfs_requested": len(etf_bars or {}),
        "etfs_measured": 0,
        "etfs_skipped_short_series": 0,
        "etfs_skipped_unnamed": 0,
        "etfs_skipped_no_session_overlap": 0,
        "benchmark_bars": 0,
        "bars_consumed": 0,
        "observations": 0,
        "session": "",
    }
    result = DetectionResult(coverage=coverage)

    if not benchmark_bars:
        # No benchmark means no spread, and a bare ETF move is not a divergence.
        # Missing data is uncertainty, never confirmation (plan.md sec 5).
        coverage["skipped_reason"] = "no benchmark bars"
        return result

    session = _session_of(benchmark_bars[0])
    bench = _session_bars(benchmark_bars, session)
    coverage["session"] = session
    coverage["benchmark_bars"] = len(bench)
    if len(bench) < min(min_bars, len(benchmark_bars)):
        coverage["skipped_reason"] = "benchmark session too short"
        return result
    bench_moves = _moves_from_open(bench)

    for etf, raw_bars in sorted((etf_bars or {}).items()):
        if not str(etf or "").strip():
            # An unknown sector excludes rather than being admitted unlabelled.
            coverage["etfs_skipped_unnamed"] += 1
            continue
        bars = _session_bars(raw_bars, session)
        if not bars:
            coverage["etfs_skipped_no_session_overlap"] += 1
            continue
        if len(bars) < min_bars:
            coverage["etfs_skipped_short_series"] += 1
            continue
        coverage["etfs_measured"] += 1
        moves = _moves_from_open(bars)
        span = min(len(moves), len(bench_moves))
        coverage["bars_consumed"] += span

        run = 0
        observation: CohortObservation | None = None
        max_abs = 0.0
        last_spread = 0.0
        qualifying = 0
        for index in range(span):
            etf_move, bench_move = moves[index], bench_moves[index]
            if etf_move is None or bench_move is None:
                run = 0
                continue
            spread = etf_move - bench_move
            last_spread = spread
            max_abs = max(max_abs, abs(spread))
            if abs(spread) < threshold:
                run = 0
                continue
            run += 1
            qualifying += 1
            if run >= persistence and observation is None:
                stamp = _stamp(bars[index])
                observation = CohortObservation(
                    session=session,
                    etf=etf,
                    direction="short" if spread < 0 else "long",
                    first_fire_bar_index=index,
                    first_fire_at=stamp.isoformat() if stamp else "",
                    spread_at_fire_pct=spread,
                    max_abs_spread_pct=abs(spread),
                    final_spread_pct=spread,
                    qualifying_bars=run,
                )
        if observation is not None:
            result.observations.append(
                CohortObservation(
                    session=observation.session,
                    etf=observation.etf,
                    direction=observation.direction,
                    first_fire_bar_index=observation.first_fire_bar_index,
                    first_fire_at=observation.first_fire_at,
                    spread_at_fire_pct=observation.spread_at_fire_pct,
                    max_abs_spread_pct=max_abs,
                    final_spread_pct=last_spread,
                    qualifying_bars=qualifying,
                )
            )
    coverage["observations"] = len(result.observations)
    return result


# ---------------------------------------------------------------------------
# member entry timing - the archetype, called rather than restated
# ---------------------------------------------------------------------------
def member_entry(
    *,
    bars: Sequence[Mapping[str, Any]],
    prior_day_low: float | None,
    config: Mapping[str, Any],
    symbol: str = "",
) -> MemberEntry | None:
    """First point-in-time entry inside a flagged cohort, or ``None``.

    Every field is read from bars strictly at or before the decision bar, and
    the fill is the NEXT bar's open, which is the first price actually
    obtainable. ``prior_day_low`` of ``None`` returns ``None``: an unmeasurable
    prior session is not a satisfied condition.
    """
    if prior_day_low is None or not bars:
        return None
    try:
        from chart_snapshot import session_vwap_series
    except Exception:  # pragma: no cover - import guard only
        logging.warning("session_vwap_series unavailable; no member entry measured.")
        return None

    opening_bars = int(config.get("opening_high_bars", 3))
    first_hour = int(config.get("first_hour_bars", 12))
    lookback = int(config.get("swing_stop_lookback_bars", 6))
    window = config.get("entry_window_et") or ["10:00", "11:30"]

    parsed = [dict(bar, _dt=_stamp(bar)) for bar in bars]
    parsed = [bar for bar in parsed if bar["_dt"] is not None]
    if len(parsed) < first_hour + 2:
        return None

    lows = [_float(bar, "low") for bar in parsed[:first_hour]]
    if not any(low is not None and low < prior_day_low for low in lows):
        return None  # the prior-day low was never broken in the first hour

    vwap = session_vwap_series([dict(bar, dt=bar["_dt"]) for bar in parsed])["vwap"]

    for index in range(1, len(parsed) - 1):
        clock = parsed[index]["_dt"].strftime("%H:%M")
        if clock < window[0]:
            continue
        if clock > window[1]:
            break
        so_far = parsed[: index + 1]
        highs = [_float(bar, "high") or float("-inf") for bar in so_far]
        # Point-in-time: the high of the session SO FAR, never the whole day's.
        if max(highs[:opening_bars], default=float("-inf")) < max(
            highs[opening_bars:], default=float("-inf")
        ):
            continue
        close = _float(parsed[index], "close")
        prior_low = _float(parsed[index - 1], "low")
        reference = vwap[index] if index < len(vwap) else None
        if close is None or prior_low is None or reference is None:
            continue
        if not (close < reference and close < prior_low):
            continue
        fill = _float(parsed[index + 1], "open")
        if fill is None or fill <= 0:
            continue
        stop_source = [
            _float(bar, "high") for bar in parsed[max(0, index - lookback + 1) : index + 1]
        ]
        stop = max([value for value in stop_source if value is not None], default=None)
        if stop is None or stop <= fill:
            continue
        return MemberEntry(
            symbol=str(symbol or "").upper(),
            entry_time_et=clock,
            fill=fill,
            stop=stop,
            risk_pct=(stop - fill) / fill * 100.0,
        )
    return None


# ---------------------------------------------------------------------------
# shadow output
# ---------------------------------------------------------------------------
def default_shadow_path() -> Path:
    try:
        from project_paths import get_diagnostics_dir

        base = Path(get_diagnostics_dir())
    except Exception:  # pragma: no cover - only when project_paths is absent
        base = Path.home() / ".tradingbotv3" / "diagnostics"
    return base / "shadow_evidence" / "sector_cohort" / "sector_cohort_shadow.jsonl"


def write_shadow_rows(
    result: DetectionResult,
    *,
    path: Path | None = None,
    config: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> int:
    """Append this run's coverage row and any observations. Returns rows written.

    A quiet run still writes its coverage row. Without it a calm market and a
    dead collector look identical in the log, and the second one would go
    unnoticed for as long as the first is plausible.
    """
    config = dict(config or resolve_config())
    target = Path(path) if path is not None else default_shadow_path()
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    identity = config_hash(config)
    rows: list[dict] = [
        {
            "schema": SHADOW_SCHEMA,
            "kind": "coverage",
            "ts": stamp,
            "config_version": config.get("version"),
            "config_hash": identity,
            **result.coverage,
        }
    ]
    rows += [
        {
            "schema": SHADOW_SCHEMA,
            "kind": "observation",
            "ts": stamp,
            "config_version": config.get("version"),
            "config_hash": identity,
            **observation.as_row(),
        }
        for observation in result.observations
    ]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    except OSError:
        logging.exception("sector cohort shadow rows not written (shadow-only; nothing else affected).")
        return 0
    return len(rows)


# ---------------------------------------------------------------------------
# the single-flight owner
# ---------------------------------------------------------------------------
_RUN_LOCK = threading.Lock()


def run_shadow_pass(
    *,
    config: Mapping[str, Any] | None = None,
    fetch=None,
    path: Path | None = None,
) -> DetectionResult | None:
    """One owner, one run at a time. ``None`` when disabled or already running.

    ``fetch(symbols) -> {symbol: bars}`` is injected so the detection logic is
    testable without a network and so the vendor stays one seam wide. The
    default fetcher is batched yfinance - the M5 Strength Board's template -
    and spends no IB pacing budget.
    """
    config = dict(config or resolve_config())
    if not config.get("enabled"):
        return None
    if not _RUN_LOCK.acquire(blocking=False):
        logging.info("sector cohort pass already running; skipping this tick.")
        return None
    try:
        symbols = list(config.get("etfs") or [])
        benchmark = str(config.get("benchmark") or "SPY")
        fetcher = fetch or _fetch_batched_m5
        series = fetcher([benchmark, *symbols])
        result = run_detection(
            benchmark_bars=series.get(benchmark) or [],
            etf_bars={sym: series.get(sym) or [] for sym in symbols},
            config=config,
        )
        write_shadow_rows(result, path=path, config=config)
        return result
    finally:
        _RUN_LOCK.release()


def _fetch_batched_m5(symbols: Sequence[str]) -> dict[str, list[dict]]:
    """Batched yfinance M5 bars. Zero IB traffic (the Strength Board template)."""
    import yfinance as yf

    frame = yf.download(
        list(symbols),
        period="1d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, list[dict]] = {}
    for symbol in symbols:
        try:
            bars = frame[symbol].dropna()
        except Exception:
            continue
        out[symbol] = [
            {
                "dt": stamp.to_pydatetime(),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "volume": float(row.Volume),
            }
            for stamp, row in bars.iterrows()
        ]
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one sector-cohort shadow pass")
    parser.add_argument("--enable", action="store_true", help="override the shipped-off switch")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = resolve_config({"enabled": True} if args.enable else None)
    result = run_shadow_pass(config=config, path=args.out)
    if result is None:
        print("sector cohort watch is disabled (pass --enable for a one-off shadow pass).")
        return 0
    print(
        f"session {result.coverage.get('session')}: "
        f"{result.coverage.get('etfs_measured')} ETFs measured, "
        f"{len(result.observations)} cohort observation(s)."
    )
    for observation in result.observations:
        print(f"  {observation.etf} {observation.direction} from {observation.first_fire_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
