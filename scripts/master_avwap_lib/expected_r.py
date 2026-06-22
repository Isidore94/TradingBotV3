"""Expected-R ranking engine for Master AVWAP priority setups.

This module converts the existing static "quality points" score into an
expected R-multiple, then blends that prior with the live, regime-conditioned
realized R that the setup tracker has been measuring.  The blend is the
mechanism that lets *what is actually working right now* lead the ranking
instead of the hand-set signal weights in ``FAVORITE_CURRENT_SIGNALS``.

Design goals:
  * Pure functions, no IB / network / pandas dependency, so the ranking math
    is fully unit-testable in isolation.
  * Dict-based config matching the house ``DEFAULT_*_CONFIG`` style so the
    anchors and shrinkage constants are tunable without code changes.
  * "Tracker leads" by default: a small shrinkage constant means even a modest
    sample of recent closed trades pulls the estimate strongly toward realized
    performance, while still shrinking back to the static prior when the sample
    is tiny (so a single +4R fluke cannot hijack the top of the list).

The headline number a trader ranks off of is ``expected_r`` (an estimate of the
R-multiple the setup is worth in the current tape), with ``rank_score`` applying
a freshness decay so stale signals cannot outrank fresh ones on points alone.
"""

from __future__ import annotations

import copy
from statistics import median

# ``shrinkage_k`` controls how many closed samples it takes before realized
# performance carries half the weight (weight = n / (n + k)).  Smaller k =
# tracker leads harder.  Anchors map static quality points -> a prior expected
# R; they are intentionally conservative (a bare-minimum favorite is only
# slightly positive in expectation before the tracker has its say).
DEFAULT_EXPECTED_R_CONFIG: dict = {
    "mode": "tracker_leads",
    "shrinkage_k": {
        "tracker_leads": 3.0,
        "balanced": 8.0,
        "conservative": 20.0,
    },
    "realized_r_clip": 4.0,
    # (quality_points, prior_expected_r) ascending in points; linearly
    # interpolated, clamped flat beyond the ends.  100 points is the existing
    # PRIORITY_FAVORITE_SETUP_MIN_SCORE threshold.
    "prior_anchors": [
        [60.0, -0.20],
        [100.0, 0.30],
        [140.0, 0.70],
        [180.0, 1.05],
    ],
    "freshness": {
        "full_days": 2,        # <= this many days since the signal: factor 1.0
        "decay_per_day": 0.08,  # subtracted per extra day beyond full_days
        "min_factor": 0.40,
    },
    # Never let realized performance fully erase the prior; keeps a sliver of
    # structural setup quality in the estimate even with a large sample.
    "max_blend_weight": 0.92,
}


def _coerce_float(value, default=None):
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    return result


def resolved_config(config: dict | None = None) -> dict:
    """Return a config dict with defaults filled in for any missing keys."""

    if not isinstance(config, dict):
        return copy.deepcopy(DEFAULT_EXPECTED_R_CONFIG)
    merged = copy.deepcopy(DEFAULT_EXPECTED_R_CONFIG)
    for key, value in config.items():
        if key == "freshness" and isinstance(value, dict):
            merged["freshness"].update(value)
        elif key == "shrinkage_k" and isinstance(value, dict):
            merged["shrinkage_k"].update(value)
        else:
            merged[key] = value
    return merged


def _interp_anchors(x: float, anchors: list) -> float:
    """Piecewise-linear interpolation across (x, y) anchors, clamped at ends."""

    points = sorted(
        (
            (_coerce_float(pair[0]), _coerce_float(pair[1]))
            for pair in anchors
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ),
        key=lambda item: (item[0] if item[0] is not None else 0.0),
    )
    points = [(px, py) for px, py in points if px is not None and py is not None]
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            frac = (x - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return points[-1][1]


def quality_points_to_prior_r(points, config: dict | None = None) -> float:
    """Map static quality points to a prior expected R-multiple."""

    cfg = resolved_config(config)
    value = _coerce_float(points, default=0.0)
    return float(_interp_anchors(value, cfg.get("prior_anchors", [])))


def shrinkage_weight(closed_samples, config: dict | None = None) -> float:
    """Weight on realized R given the number of closed tracker samples.

    weight = n / (n + k), clamped to ``max_blend_weight``.  k is selected by the
    config ``mode`` ("tracker_leads" uses the smallest k).
    """

    cfg = resolved_config(config)
    n = _coerce_float(closed_samples, default=0.0) or 0.0
    if n <= 0:
        return 0.0
    mode = str(cfg.get("mode") or "tracker_leads")
    k_table = cfg.get("shrinkage_k") or {}
    k = _coerce_float(k_table.get(mode), default=None)
    if k is None:
        k = _coerce_float(k_table.get("tracker_leads"), default=3.0)
    k = max(0.0, float(k))
    weight = n / (n + k) if (n + k) > 0 else 0.0
    return float(min(weight, float(cfg.get("max_blend_weight", 0.92))))


def freshness_factor(days_since_signal, config: dict | None = None) -> float:
    """Decay multiplier (<=1.0) based on how stale the trigger is."""

    cfg = resolved_config(config)
    fresh = cfg.get("freshness", {})
    days = _coerce_float(days_since_signal, default=0.0) or 0.0
    if days < 0:
        days = 0.0
    full_days = _coerce_float(fresh.get("full_days"), default=2.0) or 0.0
    decay = _coerce_float(fresh.get("decay_per_day"), default=0.08) or 0.0
    min_factor = _coerce_float(fresh.get("min_factor"), default=0.4) or 0.0
    if days <= full_days:
        return 1.0
    factor = 1.0 - decay * (days - full_days)
    return float(max(min_factor, factor))


def blend_expected_r(
    prior_r,
    realized_r,
    closed_samples,
    *,
    config: dict | None = None,
) -> dict:
    """Shrinkage blend of the static prior R and realized tracker R.

    Returns a dict with the components so callers can surface *why* a setup is
    ranked where it is (prior vs realized vs sample weight).
    """

    cfg = resolved_config(config)
    prior = _coerce_float(prior_r, default=0.0) or 0.0
    realized = _coerce_float(realized_r, default=None)
    n = int(_coerce_float(closed_samples, default=0.0) or 0.0)

    if realized is None or n <= 0:
        return {
            "expected_r": float(prior),
            "prior_r": float(prior),
            "realized_r": None,
            "closed_samples": max(0, n),
            "blend_weight": 0.0,
        }

    clip = abs(_coerce_float(cfg.get("realized_r_clip"), default=4.0) or 4.0)
    realized = max(-clip, min(clip, realized))
    weight = shrinkage_weight(n, cfg)
    expected = weight * realized + (1.0 - weight) * prior
    return {
        "expected_r": float(expected),
        "prior_r": float(prior),
        "realized_r": float(realized),
        "closed_samples": n,
        "blend_weight": float(weight),
    }


def compute_expected_r(
    *,
    quality_points,
    realized_r=None,
    closed_samples=0,
    days_since_signal=0,
    config: dict | None = None,
) -> dict:
    """Full Expected-R computation for a single setup.

    ``rank_score`` is the value to sort the priority list by: it equals
    ``expected_r`` decayed by freshness, but only when the expectation is
    positive (freshness must never make a negative-expectancy setup look better
    by pulling it toward zero).
    """

    cfg = resolved_config(config)
    prior = quality_points_to_prior_r(quality_points, cfg)
    blended = blend_expected_r(prior, realized_r, closed_samples, config=cfg)
    fresh = freshness_factor(days_since_signal, cfg)
    expected = blended["expected_r"]
    rank_score = expected * fresh if expected > 0 else expected
    result = dict(blended)
    result["freshness_factor"] = float(fresh)
    result["rank_score"] = float(rank_score)
    result["quality_points"] = _coerce_float(quality_points, default=0.0)
    return result


# ---------------------------------------------------------------------------
# Prior-anchor calibration
#
# The default points->R anchors are a sensible guess.  Calibration replaces them
# with anchors fitted from the trader's own closed setups: "setups that scored
# ~X static quality points realized ~Y R."  Fitted anchors are shrunk toward the
# defaults by per-bin confidence (thin bins barely move) and forced monotonic
# non-decreasing via isotonic regression, so a noisy mid bucket can't invert the
# curve.  Falls back to defaults when there isn't enough closed history.
# ---------------------------------------------------------------------------
DEFAULT_CALIBRATION: dict = {
    "min_total_samples": 30,
    "min_bin_samples": 6,
    "num_bins": 4,
    "r_clip": 4.0,
    # Per-bin shrink toward the default-anchor curve: weight = n / (n + k).
    "shrink_k": 12.0,
}


def _isotonic_non_decreasing(values: list, weights: list) -> list:
    """Pool-Adjacent-Violators fit producing a non-decreasing sequence.

    Returns one fitted value per input, weighted by ``weights``.  This is the
    statistically correct way to enforce monotonicity without the upward bias of
    a one-sided cumulative-max clamp.
    """

    blocks: list[list[float]] = []  # each block: [mean, total_weight, count]
    for value, weight in zip(values, weights):
        weight = max(float(weight), 1e-9)
        blocks.append([float(value), weight, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            mean2, weight2, count2 = blocks.pop()
            mean1, weight1, count1 = blocks.pop()
            total_weight = weight1 + weight2
            blocks.append(
                [
                    (mean1 * weight1 + mean2 * weight2) / total_weight,
                    total_weight,
                    count1 + count2,
                ]
            )
    fitted: list[float] = []
    for mean, _weight, count in blocks:
        fitted.extend([mean] * count)
    return fitted


def _quantile_bins(sorted_pairs: list, num_bins: int, min_bin: int) -> list:
    """Split point-sorted (points, r) pairs into ~equal-count bins, merging any
    bin smaller than ``min_bin`` into a neighbour so every bin is well-sampled."""

    n = len(sorted_pairs)
    num_bins = max(2, int(num_bins))
    size = max(1, n // num_bins)
    raw = [sorted_pairs[i * size : (i + 1) * size] for i in range(num_bins)]
    consumed = num_bins * size
    if consumed < n:
        raw[-1].extend(sorted_pairs[consumed:])

    merged: list[list] = []
    for chunk in raw:
        if not chunk:
            continue
        if merged and len(chunk) < int(min_bin):
            merged[-1].extend(chunk)
        else:
            merged.append(list(chunk))
    if len(merged) >= 2 and len(merged[0]) < int(min_bin):
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return [chunk for chunk in merged if chunk]


def calibrate_prior_anchors(
    samples,
    *,
    config: dict | None = None,
    calibration: dict | None = None,
) -> dict:
    """Fit points->R prior anchors from (quality_points, realized_r) samples.

    Returns a dict with the new ``anchors`` (a list of ``[points, r]`` ready to
    drop into config), ``num_samples``, ``used_default`` and a human-readable
    ``note`` plus per-bin diagnostics.  When data is thin it returns the default
    anchors unchanged so calibration can never make ranking worse.
    """

    cfg = resolved_config(config)
    cal = {**DEFAULT_CALIBRATION, **(calibration or {})}
    default_anchors = [list(pair) for pair in cfg.get("prior_anchors", [])]
    clip = abs(_coerce_float(cal.get("r_clip"), default=4.0) or 4.0)

    pairs: list[tuple[float, float]] = []
    for item in samples or []:
        try:
            pts, r = item
        except (TypeError, ValueError):
            continue
        p = _coerce_float(pts)
        rr = _coerce_float(r)
        if p is None or rr is None:
            continue
        pairs.append((float(p), max(-clip, min(clip, float(rr)))))

    min_total = int(cal.get("min_total_samples", 30) or 30)
    if len(pairs) < min_total:
        return {
            "anchors": default_anchors,
            "num_samples": len(pairs),
            "used_default": True,
            "bins": [],
            "note": f"insufficient closed samples ({len(pairs)}<{min_total}); kept default anchors",
        }

    pairs.sort(key=lambda pair: pair[0])
    bins = _quantile_bins(pairs, int(cal.get("num_bins", 4) or 4), int(cal.get("min_bin_samples", 6) or 6))
    if len(bins) < 2:
        return {
            "anchors": default_anchors,
            "num_samples": len(pairs),
            "used_default": True,
            "bins": [],
            "note": "could not form >=2 well-sampled bins; kept default anchors",
        }

    shrink_k = max(0.0, float(_coerce_float(cal.get("shrink_k"), default=12.0) or 12.0))
    bin_points: list[float] = []
    bin_weights: list[float] = []
    bin_raw_r: list[float] = []
    bin_shrunk_r: list[float] = []
    for chunk in bins:
        pts_values = [p for p, _ in chunk]
        r_values = [r for _, r in chunk]
        rep_points = float(median(pts_values))
        raw_r = float(median(r_values))
        n = len(chunk)
        default_r = _interp_anchors(rep_points, default_anchors)
        weight = n / (n + shrink_k) if (n + shrink_k) > 0 else 0.0
        shrunk_r = weight * raw_r + (1.0 - weight) * default_r
        bin_points.append(rep_points)
        bin_weights.append(float(n))
        bin_raw_r.append(raw_r)
        bin_shrunk_r.append(shrunk_r)

    fitted_r = _isotonic_non_decreasing(bin_shrunk_r, bin_weights)
    anchors = [[round(p, 1), round(r, 3)] for p, r in zip(bin_points, fitted_r)]

    bin_diagnostics = [
        {
            "points": round(bin_points[i], 1),
            "samples": int(bin_weights[i]),
            "median_realized_r": round(bin_raw_r[i], 3),
            "shrunk_r": round(bin_shrunk_r[i], 3),
            "fitted_r": round(fitted_r[i], 3),
        }
        for i in range(len(bin_points))
    ]
    return {
        "anchors": anchors,
        "num_samples": len(pairs),
        "used_default": False,
        "bins": bin_diagnostics,
        "note": f"calibrated from {len(pairs)} closed samples across {len(anchors)} bins",
    }
