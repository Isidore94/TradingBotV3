"""Side-symmetric relative-strength engine (plan.md Packet C, sec 16.2/16.6-16.8).

Pure computation: callers fetch bars; this module aligns exact timestamps,
computes multi-window aligned features in one pass, and ranks a candidate
cohort with a transparent component/percentile composite. Long (RS) and short
(RW) share one signed coordinate system (`side_sign`); no `abs()` where sign
carries meaning, so mirrored input with the opposite side must produce
identical rankings (tested).

Raw percent excess is never the only measure: residuals are beta-adjusted and
volatility-normalized, and every component is stored on the result so ranking
stays explainable and calibratable out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from market_state import M5Bar

ENGINE_VERSION = "rs_engine_v1"

DEFAULT_WINDOWS_MINUTES = (5, 15, 30, 60)


@dataclass(frozen=True)
class RelativeStrengthConfig:
    version: str = ENGINE_VERSION
    windows_minutes: tuple[int, ...] = DEFAULT_WINDOWS_MINUTES
    min_coverage: float = 0.8
    volatility_floor_pct: float = 0.05  # per-window normalization floor
    # component weights (sec 16.8 hypotheses; calibrate out of sample)
    weight_residual: float = 0.30
    weight_giveback: float = 0.20
    weight_persistence: float = 0.15
    weight_sector_residual: float = 0.10
    weight_volume_quality: float = 0.10
    weight_setup_quality: float = 0.10
    weight_trigger_freshness: float = 0.05
    # explicit penalties (composite points, subtracted after the weighted mix)
    penalty_extension_per_atr: float = 8.0
    penalty_stale_data: float = 25.0
    penalty_imminent_earnings: float = 10.0
    earnings_risk_days: int = 2
    max_extension_atr: float = 2.0
    # tier thresholds (cross-sectional percentile of the composite)
    defiant_percentile: float = 0.80
    holding_percentile: float = 0.55


@dataclass(frozen=True)
class AlignedWindowFeatures:
    window_minutes: int
    coverage: float
    aligned_bar_count: int
    first_ts: datetime | None
    last_ts: datetime | None
    stock_return_pct: float
    spy_return_pct: float
    sector_return_pct: float | None
    aligned_residual_pct: float          # side_sign * (stock - beta*spy)
    aligned_sector_residual_pct: float | None
    volatility_adjusted_residual: float  # residual / vol scale
    ok: bool


@dataclass
class CandidateInput:
    symbol: str
    side_sign: int  # +1 long/RS, -1 short/RW
    stock_bars: list[M5Bar]
    sector_bars: list[M5Bar] | None = None
    beta: float = 1.0
    volume_quality: float | None = None        # 0..1 relative-volume quality
    setup_quality: float | None = None         # 0..1 higher-timeframe quality
    trigger_proximity: float | None = None     # 0..1, 1 = at trigger and fresh
    extension_atr: float | None = None         # distance beyond valid entry ref
    earnings_within_days: int | None = None
    structure_ok: bool = True


@dataclass
class CandidateRank:
    symbol: str
    side_sign: int
    tier: str
    composite: float
    percentile: float
    components: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    windows: dict[int, AlignedWindowFeatures] = field(default_factory=dict)
    coverage: float = 0.0
    data_ok: bool = True
    engine_version: str = ENGINE_VERSION


# ---------------------------------------------------------------------------
# aligned window features
# ---------------------------------------------------------------------------

def _closes_by_ts(bars: list[M5Bar]) -> dict[datetime, float]:
    return {b.ts: b.close for b in bars if b.complete}


def compute_window_features(
    stock_bars: list[M5Bar],
    spy_bars: list[M5Bar],
    *,
    window_minutes: int,
    side_sign: int,
    beta: float = 1.0,
    sector_bars: list[M5Bar] | None = None,
    min_coverage: float = 0.8,
    volatility_floor_pct: float = 0.05,
) -> AlignedWindowFeatures:
    """Features over the EXACT common timestamps of stock and SPY.

    The window ends at the latest common timestamp — a symbol missing recent
    bars is never credited/blamed for an index move it has no data for.
    The window is endpoint-inclusive: a bar exactly ``window_minutes`` back is
    IN the window, so an N-minute window on 5m bars holds N/5 + 1 bars and the
    measured return spans the full labeled duration (a 5-minute window is two
    bars, one bar-to-bar return).
    """
    stock = _closes_by_ts(stock_bars)
    spy = _closes_by_ts(spy_bars)
    common = sorted(set(stock) & set(spy))
    empty = AlignedWindowFeatures(
        window_minutes=window_minutes,
        coverage=0.0,
        aligned_bar_count=0,
        first_ts=None,
        last_ts=None,
        stock_return_pct=0.0,
        spy_return_pct=0.0,
        sector_return_pct=None,
        aligned_residual_pct=0.0,
        aligned_sector_residual_pct=None,
        volatility_adjusted_residual=0.0,
        ok=False,
    )
    if not common:
        return empty
    window_end = common[-1]
    expected_bars = max(2, window_minutes // 5 + 1)
    in_window = [ts for ts in common if (window_end - ts).total_seconds() <= window_minutes * 60]
    if len(in_window) < 2:
        return empty
    coverage = min(1.0, len(in_window) / expected_bars)
    first_ts, last_ts = in_window[0], in_window[-1]

    def pct(series: dict[datetime, float]) -> float:
        start, end = series[first_ts], series[last_ts]
        return (end - start) / start * 100.0 if start else 0.0

    stock_ret = pct(stock)
    spy_ret = pct(spy)
    residual = side_sign * (stock_ret - beta * spy_ret)

    sector_ret = None
    sector_residual = None
    if sector_bars:
        sector = _closes_by_ts(sector_bars)
        if first_ts in sector and last_ts in sector:
            sector_ret = pct(sector)
            sector_residual = side_sign * (stock_ret - sector_ret)

    # Volatility scale: std of the stock's aligned per-bar returns over the
    # window, in percent, floored so quiet tape cannot divide by ~zero.
    rets = []
    for prev, cur in zip(in_window, in_window[1:]):
        if stock[prev]:
            rets.append((stock[cur] - stock[prev]) / stock[prev] * 100.0)
    if len(rets) >= 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        vol_scale = max(volatility_floor_pct, (var ** 0.5) * (len(rets) ** 0.5))
    else:
        vol_scale = volatility_floor_pct
    vol_adj = residual / vol_scale

    return AlignedWindowFeatures(
        window_minutes=window_minutes,
        coverage=coverage,
        aligned_bar_count=len(in_window),
        first_ts=first_ts,
        last_ts=last_ts,
        stock_return_pct=stock_ret,
        spy_return_pct=spy_ret,
        sector_return_pct=sector_ret,
        aligned_residual_pct=residual,
        aligned_sector_residual_pct=sector_residual,
        volatility_adjusted_residual=vol_adj,
        ok=coverage >= min_coverage,
    )


# ---------------------------------------------------------------------------
# cohort ranking
# ---------------------------------------------------------------------------

def _percentile_ranks(values: list[float]) -> list[float]:
    """Cross-sectional percentile (0..1) of each value within the cohort."""
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    order = sorted(values)
    n = len(values)
    ranks = []
    for v in values:
        below = sum(1 for o in order if o < v)
        equal = sum(1 for o in order if o == v)
        ranks.append((below + 0.5 * equal) / n)
    return ranks


class RelativeStrengthEngine:
    """Rank a candidate cohort against SPY over aligned multi-minute windows."""

    def __init__(self, config: RelativeStrengthConfig | None = None) -> None:
        self.config = config or RelativeStrengthConfig()

    def rank(
        self,
        spy_bars: list[M5Bar],
        candidates: list[CandidateInput],
        *,
        reference_window_minutes: int = 30,
    ) -> list[CandidateRank]:
        cfg = self.config
        raw: list[dict] = []
        for cand in candidates:
            windows: dict[int, AlignedWindowFeatures] = {}
            for minutes in cfg.windows_minutes:
                windows[minutes] = compute_window_features(
                    cand.stock_bars,
                    spy_bars,
                    window_minutes=minutes,
                    side_sign=cand.side_sign,
                    beta=cand.beta,
                    sector_bars=cand.sector_bars,
                    min_coverage=cfg.min_coverage,
                    volatility_floor_pct=cfg.volatility_floor_pct,
                )
            ref = windows.get(reference_window_minutes) or next(iter(windows.values()))
            persistence_values = [
                w.aligned_residual_pct for w in windows.values() if w.ok
            ]
            persistence = (
                sum(1 for v in persistence_values if v > 0) / len(persistence_values)
                if persistence_values
                else 0.0
            )
            # Giveback resistance: how much better than beta-expected the stock
            # held over the reference window (identical to the residual today;
            # kept as its own named component so an episode-anchored variant
            # can replace it without reshaping stored records).
            giveback_resistance = ref.aligned_residual_pct
            raw.append(
                {
                    "cand": cand,
                    "windows": windows,
                    "ref": ref,
                    "residual": ref.volatility_adjusted_residual,
                    "giveback": giveback_resistance,
                    "persistence": persistence,
                    "sector_residual": ref.aligned_sector_residual_pct or 0.0,
                    "volume_quality": cand.volume_quality if cand.volume_quality is not None else 0.5,
                    "setup_quality": cand.setup_quality if cand.setup_quality is not None else 0.5,
                    "trigger": cand.trigger_proximity if cand.trigger_proximity is not None else 0.0,
                }
            )

        keys = ("residual", "giveback", "persistence", "sector_residual", "volume_quality", "setup_quality", "trigger")
        percentiles = {k: _percentile_ranks([r[k] for r in raw]) for k in keys}
        weights = {
            "residual": cfg.weight_residual,
            "giveback": cfg.weight_giveback,
            "persistence": cfg.weight_persistence,
            "sector_residual": cfg.weight_sector_residual,
            "volume_quality": cfg.weight_volume_quality,
            "setup_quality": cfg.weight_setup_quality,
            "trigger": cfg.weight_trigger_freshness,
        }

        results: list[CandidateRank] = []
        for i, r in enumerate(raw):
            cand: CandidateInput = r["cand"]
            components = {k: percentiles[k][i] for k in keys}
            composite = 100.0 * sum(weights[k] * components[k] for k in keys)
            penalties: dict[str, float] = {}
            if cand.extension_atr is not None and cand.extension_atr > 0:
                penalties["extension"] = cfg.penalty_extension_per_atr * min(
                    cand.extension_atr, cfg.max_extension_atr * 2
                )
            data_ok = r["ref"].ok
            if not data_ok:
                penalties["stale_data"] = cfg.penalty_stale_data
            if (
                cand.earnings_within_days is not None
                and cand.earnings_within_days <= cfg.earnings_risk_days
            ):
                penalties["imminent_earnings"] = cfg.penalty_imminent_earnings
            composite -= sum(penalties.values())
            results.append(
                CandidateRank(
                    symbol=cand.symbol,
                    side_sign=cand.side_sign,
                    tier="",
                    composite=composite,
                    percentile=0.0,
                    components=components,
                    penalties=penalties,
                    windows=r["windows"],
                    coverage=r["ref"].coverage,
                    data_ok=data_ok,
                )
            )

        composite_percentiles = _percentile_ranks([res.composite for res in results])
        for res, pct_rank, r in zip(results, composite_percentiles, raw):
            res.percentile = pct_rank
            res.tier = self._tier(res, r)
        results.sort(key=lambda res: (-res.composite, res.symbol))
        return results

    def _tier(self, res: CandidateRank, raw: dict) -> str:
        cfg = self.config
        cand: CandidateInput = raw["cand"]
        ref: AlignedWindowFeatures = raw["ref"]
        if not cand.structure_ok:
            return "INVALID"
        if not res.data_ok:
            return "WATCHING"
        over_extended = (
            cand.extension_atr is not None and cand.extension_atr > cfg.max_extension_atr
        )
        if ref.aligned_residual_pct < 0 and raw["persistence"] < 0.5:
            return "FADING"
        if res.percentile >= cfg.defiant_percentile and ref.aligned_residual_pct > 0:
            return "WATCHING" if over_extended else "DEFIANT"
        if res.percentile >= cfg.holding_percentile and ref.aligned_residual_pct >= 0:
            return "WATCHING" if over_extended else "HOLDING"
        return "WATCHING"


def mirror_candidate(cand: CandidateInput, pivot: float) -> CandidateInput:
    """Mirror a candidate's bars around `pivot` and flip its side."""
    from market_state import mirror_bar

    return CandidateInput(
        symbol=cand.symbol,
        side_sign=-cand.side_sign,
        stock_bars=[mirror_bar(b, pivot) for b in cand.stock_bars],
        sector_bars=[mirror_bar(b, pivot) for b in cand.sector_bars] if cand.sector_bars else None,
        beta=cand.beta,
        volume_quality=cand.volume_quality,
        setup_quality=cand.setup_quality,
        trigger_proximity=cand.trigger_proximity,
        extension_atr=cand.extension_atr,
        earnings_within_days=cand.earnings_within_days,
        structure_ok=cand.structure_ok,
    )
