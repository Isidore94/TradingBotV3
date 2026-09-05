"""Tier-1 feature snapshots (plan Phase 5; frozen columns in sec 7.1).

The rule that shapes this module: **the champion's computation is the
computation.** `calc_anchored_vwap_bands` is called, never re-derived - its
running-deviation sigma is frozen (decision 0008) and every existing consumer
is calibrated to it, so a second implementation that agreed today would be a
silent trap tomorrow. The same applies to the D1 indicator grid
(`compute_indicator_frame`) and to the intraday session-VWAP band math
(`_calculate_vwap_bands`, which touches no instance state and is therefore
callable unbound rather than by standing up the whole BounceBot app).

What is genuinely new here is the favorite-zone block of sec 6.2. Those columns
are frozen in the schema but the plan states only what they measure, not their
arithmetic, so each carries a stated v1 definition under
``feature_set_version``. Two of them - ``first_dev_touch_order`` and
``band1_rejection_strength`` - are on the confirm-or-amend list for the trader;
a later definition is a version bump plus additive rows, never a rewrite of
history.

Production context values (RVOL, RS/RW, group RS, market internals, the chop
veto, pullback count) are **passed in from wrapped production evidence, never
recomputed**. Recomputing a champion quantity with a second formula is how two
numbers with one name start disagreeing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

try:  # package import
    from . import exchange_calendar as xcal
    from .manifest import utc_now
    from .schemas import SCHEMA_VERSION, anchor_instance_id
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import SCHEMA_VERSION, anchor_instance_id  # type: ignore
    from store import ResearchStore  # type: ignore

#: Bump when any definition below changes; old rows keep their version.
#:
#: ``tier1_v2`` (M4.1, 2026-09-05) added the challenger AVWAP band family
#: (``avwap_variant_*``). No ``tier1_v1`` definition changed, and no
#: ``tier1_v1`` row is rewritten: the dataset identity carries the version, so
#: the two shapes coexist and a rebuild supersedes rather than editing.
FEATURE_SET_VERSION = "tier1_v2"
#: The frozen running-deviation sigma variant (decision 0008).
AVWAP_FORMULA_VERSION = "running_deviation_v1"

ATR_LENGTH = 14  # the schema's declared atr14 column
VWAP_ALGORITHM_STANDARD = "STANDARD"

#: D1 history a tier-1 snapshot must see before it may be computed.
#:
#: The champion `compute_indicator_frame` uses `rolling(period)` with pandas'
#: default ``min_periods``, so ``sma100``/``sma200`` are silently null unless
#: the frame carries 100/200 completed sessions, and its EMAs (``adjust=False``)
#: are seeded at the frame's *first* bar - a truncated frame therefore produces
#: different numbers under the same column names (BD-33's failure mode).
#: 250 sessions covers ``sma200`` with margin and drives the EMA-21 seed error
#: below float tolerance (``(1 - 2/22) ** 250`` is about 5e-11); it also exceeds
#: the champion's own deepest D1 fetch (``PRIORITY_SMA_LOOKBACK_DAYS`` = 320
#: calendar days, about 220 sessions).
DAILY_HISTORY_MIN_SESSIONS = 250
#: Hard stop on the year walk, so a sparse lake cannot read the whole store.
DAILY_HISTORY_MAX_YEARS = 5

ANCHOR_TYPE_CURRENT = "EARNINGS_CURRENT"
ANCHOR_TYPE_PREVIOUS = "EARNINGS_PREVIOUS"

# --- anchor knowledge (Q2.1, BD-99) ---------------------------------------
#: The anchor was in the lake BEFORE the session it is used for: the desk could
#: have known it that day.
ANCHOR_KNOWLEDGE_OBSERVED = "observed"
#: The anchor's ``system_from`` is AFTER the session: it was imported later and
#: the row is research evidence, never point-in-time evidence for a promotion
#: gate (plan.md sec 7, BD-99).
ANCHOR_KNOWLEDGE_RECONSTRUCTED = "reconstructed"
#: Stored on a row computed with no anchor at all. The column exists, so this
#: is a statement, not a silence.
ANCHOR_KNOWLEDGE_UNANCHORED = ""
#: Reader-side buckets. A row written before the column existed reads NULL and
#: is ``legacy`` - never assumed observed; a row that had no anchor is ``none``;
#: a value this vocabulary does not recognise is ``unknown`` and borrows no
#: other bucket's meaning.
ANCHOR_KNOWLEDGE_LEGACY = "legacy"
ANCHOR_KNOWLEDGE_NONE = "none"
ANCHOR_KNOWLEDGE_UNKNOWN = "unknown"
ANCHOR_KNOWLEDGE_BUCKETS = (
    ANCHOR_KNOWLEDGE_OBSERVED,
    ANCHOR_KNOWLEDGE_RECONSTRUCTED,
    ANCHOR_KNOWLEDGE_NONE,
    ANCHOR_KNOWLEDGE_LEGACY,
    ANCHOR_KNOWLEDGE_UNKNOWN,
)


@dataclass(frozen=True)
class AnchorChoice:
    """Which anchor a session's snapshot uses, and whether it was KNOWN then.

    The bar date alone cannot answer the second question: the 2026-09-04
    earnings-anchor bridge back-filled ~2,200 anchors whose bars are months old
    and whose knowledge stamp is that night. A snapshot rebuilt for an August
    session over those rows is legitimate research evidence and would be a lie
    as point-in-time evidence, so the row says which it is.
    """

    anchor_bar_date: date
    knowledge: str = ANCHOR_KNOWLEDGE_RECONSTRUCTED


def anchor_knowledge_bucket(value) -> str:
    """The reader's bucket for a stored ``anchor_knowledge`` value.

    ``None`` means the row predates the column (Q2.1) and is ``legacy``:
    uncertainty is never read as confirmation, so it never pools with
    ``observed``. ``""`` is ``none`` - the POSITIVE statement "this row used no
    anchor" - and a value the vocabulary does not recognise is ``unknown``,
    never ``none``: a future writer's label must not silently be counted as
    "had no anchor".
    """
    if value is None:
        return ANCHOR_KNOWLEDGE_LEGACY
    text = str(value).strip()
    if text in (ANCHOR_KNOWLEDGE_OBSERVED, ANCHOR_KNOWLEDGE_RECONSTRUCTED):
        return text
    return ANCHOR_KNOWLEDGE_NONE if text == ANCHOR_KNOWLEDGE_UNANCHORED else ANCHOR_KNOWLEDGE_UNKNOWN


def _anchor_choice(value) -> AnchorChoice | None:
    """Accept a bare date or an :class:`AnchorChoice`; an unstamped date is
    ``reconstructed``, because a caller that did not state the knowledge has
    not established it."""
    if value is None:
        return None
    if isinstance(value, AnchorChoice):
        return value
    if isinstance(value, datetime):
        return AnchorChoice(value.date())
    if isinstance(value, date):
        return AnchorChoice(value)
    bar_date = getattr(value, "anchor_bar_date", None)
    if bar_date is None:
        return None
    return AnchorChoice(
        bar_date.date() if isinstance(bar_date, datetime) else bar_date,
        str(getattr(value, "knowledge", "") or ANCHOR_KNOWLEDGE_RECONSTRUCTED),
    )


def _ensure_scripts_on_path() -> None:
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parents[1])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _master_legacy():
    """The champion scanner core, imported lazily (it is a heavy module)."""
    _ensure_scripts_on_path()
    try:
        from master_avwap_lib import legacy
    except ImportError:  # pragma: no cover - packaged import
        from scripts.master_avwap_lib import legacy  # type: ignore
    return legacy


def _bounce_legacy():
    _ensure_scripts_on_path()
    try:
        from bounce_bot_lib import legacy
    except ImportError:  # pragma: no cover - packaged import
        from scripts.bounce_bot_lib import legacy  # type: ignore
    return legacy


# ---------------------------------------------------------------------------
# Wrapped champion computations - no formula lives in this module
# ---------------------------------------------------------------------------
def anchored_vwap_bands(bars, anchor_index: int = 0):
    """Call the champion ``calc_anchored_vwap_bands`` on lake bars.

    Returns ``(vwap, stdev, bands)`` exactly as the champion returns them. The
    only work done here is shaping lake rows into the frame it expects; the
    sigma math is the champion's and is never touched (plan.md sec 5).
    """
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "open": float(row.get("open")),
                "high": float(row.get("high")),
                "low": float(row.get("low")),
                "close": float(row.get("close")),
                "volume": float(row.get("volume") or 0.0),
            }
            for row in bars
        ]
    )
    if frame.empty:
        return float("nan"), float("nan"), {}
    return _master_legacy().calc_anchored_vwap_bands(frame, int(anchor_index))


def _band_variants():
    """The pure challenger formula, imported lazily.

    ``scripts/indicators/`` is a dependency-light, offline package; importing it
    at module scope would drag it into every headless job that only wants a
    champion number. It is already collected by the PyInstaller spec
    (``FIRST_PARTY_PACKAGES``) and already named in
    ``selftest.LAZY_ENGINE_MODULES``, so this importer adds no packaging trigger.
    """
    _ensure_scripts_on_path()
    try:
        from indicators import avwap_band_variants
    except ImportError:  # pragma: no cover - packaged import
        from scripts.indicators import avwap_band_variants  # type: ignore
    return avwap_band_variants


def avwap_variant_bands(bars, anchor_index: int = 0):
    """The CHALLENGER's ``(vwap, stdev, bands)`` on lake bars (M4.1).

    ``AVWAP(HLC/3) +/- k * stdev(close, 20, population)``, the OneOption /
    Option Stalker replication pinned in ``docs/AVWAP_BAND_VARIANT_STUDY.md``
    section 2b. Deliberately the same three-tuple shape as
    :func:`anchored_vwap_bands` so a caller holds both answers without adapting
    either, and deliberately a DIFFERENT function: the champion's sigma is
    frozen (decision 0008, plan.md sec 5) and nothing here touches it.

    Unmeasurable is ``None``, never zero and never a padded window - fewer than
    twenty completed closes up to the session leaves the sigma and every band
    ``None`` while the centre, which IS measurable, is still reported.
    """
    if not bars:
        return None, None, {}
    return _band_variants().oneoption_avwap_bands(list(bars), int(anchor_index))


def indicator_grid(bars):
    """The champion D1 indicator frame (EMA 8/15/21, SMA 50/100/200)."""
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "datetime": row.get("session_date"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
            }
            for row in bars
        ]
    )
    if frame.empty:
        return None
    return _master_legacy().compute_indicator_frame(frame)


def session_vwap_bands(intraday_bars, *, band_mult: float = 1.0):
    """The champion intraday VWAP + 1σ bands, called unbound.

    ``_calculate_vwap_bands`` uses no instance state, so the computation is
    reused directly instead of instantiating the BounceBot application (which
    would drag the GUI and broker stack into a headless build job).
    """
    import pandas as pd

    rows = [
        {
            "typical_price": (
                float(bar.get("open")) + float(bar.get("high")) + float(bar.get("low")) + float(bar.get("close"))
            )
            / 4.0,
            "volume": float(bar.get("volume") or 0.0),
        }
        for bar in intraday_bars
        if bar.get("close") is not None
    ]
    if not rows:
        return None, None, None
    frame = pd.DataFrame(rows)
    return _bounce_legacy().BounceBot._calculate_vwap_bands(None, frame, band_mult=band_mult)


def atr(bars, length: int = ATR_LENGTH):
    """House ATR: mean true range over ``length`` completed bars.

    Same true-range definition and simple-mean method as the champion's
    ``compute_atr_from_ohlc``; only the window differs, because the frozen
    schema column is ``atr14`` while the scanner's own constant is 20.
    """
    window = [row for row in bars][-int(length) :]
    if not window:
        return None
    ranges = []
    previous_close = None
    for row in window:
        high, low, close = row.get("high"), row.get("low"), row.get("close")
        if high is None or low is None or close is None:
            continue
        if previous_close is None:
            ranges.append(float(high) - float(low))
        else:
            ranges.append(
                max(
                    float(high) - float(low),
                    abs(float(high) - previous_close),
                    abs(float(low) - previous_close),
                )
            )
        previous_close = float(close)
    return sum(ranges) / len(ranges) if ranges else None


def ema_series(values, span: int, *, min_bars: int | None = None):
    """EMA with the champion's convention (``adjust=False``).

    ``min_bars`` reproduces the champion's own refusal to publish a barely
    seeded EMA: BounceBot computes ``ema_8/15/21`` only ``if len(today_df) >=
    span`` (`bounce_bot_lib/legacy.py`, "Calculate short EMAs (today only)")
    and leaves the level ``None`` otherwise. Without the guard the warehouse
    would store a mostly-seed number under the champion's column name.
    """
    import pandas as pd

    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    if min_bars is not None and len(numbers) < int(min_bars):
        return None
    return float(pd.Series(numbers).ewm(span=int(span), adjust=False).mean().iloc[-1])


# ---------------------------------------------------------------------------
# The sec 6.2 favorite-zone block - stated definitions, versioned
# ---------------------------------------------------------------------------
@dataclass
class FavoriteZone:
    coord: float | None = None
    residence_bars: int | None = None
    first_dev_touch_order: int | None = None
    band1_rejection_strength: float | None = None
    second_band_streak: int | None = None


def favorite_zone_block(bars, avwap: float, bands: dict) -> FavoriteZone:
    """The 1σ favorite-zone features (sec 6.2), long-side orientation.

    Definitions, v1 (``FEATURE_SET_VERSION``):

    * ``coord`` - ``(close - AVWAPE) / (UPPER_1 - AVWAPE)`` exactly as the plan
      states it. 0 sits on the anchor VWAP, 1 on the first deviation band. The
      plan says it is mirrored for shorts, so the mirroring belongs to the
      consumer that knows the side; the stored value is long-oriented.
    * ``residence_bars`` - consecutive completed bars, ending at the snapshot,
      whose close is inside [AVWAPE, UPPER_1].
    * ``first_dev_touch_order`` - how many separate touch episodes of UPPER_1
      have occurred since the anchor, where consecutive touching bars are one
      episode. 1 means the current touch is the first. **Confirm-or-amend.**
    * ``band1_rejection_strength`` - on the most recent touching bar, the share
      of that bar's range given back from the high by the close:
      ``(high - close) / (high - low)``, in [0, 1]. **Confirm-or-amend.**
    * ``second_band_streak`` - consecutive completed bars, ending at the
      snapshot, closing at or above UPPER_2 (the 2nd-deviation power hold).
    """
    zone = FavoriteZone()
    upper_1 = bands.get("UPPER_1")
    upper_2 = bands.get("UPPER_2")
    completed = [row for row in bars if row.get("close") is not None]
    if not completed or upper_1 is None or avwap is None:
        return zone
    width = float(upper_1) - float(avwap)
    last_close = float(completed[-1]["close"])
    if width:
        zone.coord = (last_close - float(avwap)) / width

    residence = 0
    for row in reversed(completed):
        close = float(row["close"])
        if float(avwap) <= close <= float(upper_1):
            residence += 1
        else:
            break
    zone.residence_bars = residence

    if upper_2 is not None:
        streak = 0
        for row in reversed(completed):
            if float(row["close"]) >= float(upper_2):
                streak += 1
            else:
                break
        zone.second_band_streak = streak

    episodes = 0
    touching_previous = False
    last_touch = None
    for row in completed:
        high = row.get("high")
        touching = high is not None and float(high) >= float(upper_1)
        if touching and not touching_previous:
            episodes += 1
        if touching:
            last_touch = row
        touching_previous = touching
    if episodes:
        zone.first_dev_touch_order = episodes
    if last_touch is not None:
        high = float(last_touch["high"])
        low = float(last_touch["low"])
        close = float(last_touch["close"])
        span = high - low
        if span > 0:
            zone.band1_rejection_strength = max(0.0, min(1.0, (high - close) / span))
    return zone


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
@dataclass
class SnapshotReport:
    dataset: str = ""
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_COMPUTE
    rows: int = 0
    symbols: int = 0
    skipped: dict = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def build_anchor_instances(
    store: ResearchStore | None,
    anchors,
    *,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "anchor_instance",
) -> SnapshotReport:
    """Publish ``anchor_instance`` rows from wrapped earnings anchor evidence.

    ``anchors`` are dicts with ``symbol``, ``anchor_type``, ``anchor_bar_date``
    and optionally ``price_basis`` / ``source`` / ``catalyst_event_id``. The
    identity is the deterministic hash in ``schemas.py``, and the rows are
    bitemporal: a corrected anchor supersedes by ``system_from``, never by
    overwriting (sec 9.5).
    """
    report = SnapshotReport(dataset="anchor_instance")
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    rows = []
    seen = set()
    for anchor in anchors or []:
        symbol = str(anchor.get("symbol") or "").strip().upper()
        anchor_type = str(anchor.get("anchor_type") or ANCHOR_TYPE_CURRENT).upper()
        bar_date = anchor.get("anchor_bar_date")
        if isinstance(bar_date, datetime):
            bar_date = bar_date.date()
        if isinstance(bar_date, str):
            try:
                bar_date = date.fromisoformat(bar_date[:10])
            except ValueError:
                bar_date = None
        if not symbol or bar_date is None:
            report.skip("INCOMPLETE_ANCHOR")
            continue
        instance_id = anchor_instance_id(symbol, anchor_type, bar_date, AVWAP_FORMULA_VERSION)
        if instance_id in seen:
            continue
        seen.add(instance_id)
        valid_from = datetime(bar_date.year, bar_date.month, bar_date.day, tzinfo=timezone.utc)
        rows.append(
            {
                "anchor_instance_id": instance_id,
                "symbol": symbol,
                "anchor_type": anchor_type,
                "anchor_bar_date": bar_date,
                "catalyst_event_id": anchor.get("catalyst_event_id"),
                "price_basis": str(anchor.get("price_basis") or "ohlc4"),
                "anchor_bar_included": bool(anchor.get("anchor_bar_included", True)),
                "formula_version": AVWAP_FORMULA_VERSION,
                "source": str(anchor.get("source") or "earnings_avwap_anchors.csv"),
                "valid_from": valid_from,
                "valid_to": None,
                # Knowledge interval: an anchor is available once observed, not
                # retroactively at its bar (sec 6.2).
                "system_from": stamp,
                "system_to": None,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    existing = {
        str(value)
        for year in {row["anchor_bar_date"].year for row in rows}
        for value in store.read_table("anchor_instance", f"year={year}", columns=["anchor_instance_id"])
        .column("anchor_instance_id")
        .to_pylist()
    }
    rows = [row for row in rows if row["anchor_instance_id"] not in existing]
    if not rows:
        report.status = "NOTHING_TO_COMPUTE"
        return report
    report.rows = store.publish("anchor_instance", rows, job_id=job_id).rows_published
    report.symbols = len({row["symbol"] for row in rows})
    return report


# ---------------------------------------------------------------------------
# Daily snapshots
# ---------------------------------------------------------------------------
def _worst_capture_mode(modes) -> str:
    order = {"LIVE": 0, "DELAYED": 1, "BACKFILL": 2, "RECONSTRUCTED": 3, "": 4}
    worst, rank = "", -1
    for mode in modes:
        value = order.get(str(mode or ""), 4)
        if value > rank:
            rank, worst = value, str(mode or "")
    return worst


def input_manifest_hash(store: ResearchStore, datasets_partitions) -> str:
    """Hash of the exact input file set, for reproducibility (sec 7.1).

    Built from the manifest's own file hashes, so re-running against the same
    sealed inputs reproduces the same value and a changed input is visible.
    """
    digest = hashlib.sha256()
    for dataset, partition in sorted(datasets_partitions):
        snapshot = store.manifest.resolve(dataset=dataset, partition=partition)
        for entry in sorted(snapshot.entries, key=lambda item: item.file_path):
            digest.update(f"{entry.file_path}:{entry.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def compute_daily_features(
    symbol: str,
    d1_rows,
    *,
    session_date: date,
    anchor_index: int | None = None,
    anchor_knowledge: str = ANCHOR_KNOWLEDGE_UNANCHORED,
    spy_regime_state: str | None = None,
    manifest_hash: str = "",
    computed_at: datetime | None = None,
    feature_set_version: str = FEATURE_SET_VERSION,
    run_id: str = "",
) -> dict | None:
    """One ``feature_snapshot_daily`` row from canonical D1 bars.

    Only bars completed on or before ``session_date`` are used - a snapshot may
    never see a bar that did not exist at its own event time.
    """
    usable = []
    for row in d1_rows:
        day = row.get("session_date")
        if isinstance(day, datetime):
            day = day.date()
        if day is None or day > session_date or not row.get("is_complete", True):
            continue
        usable.append(row)
    usable.sort(key=lambda row: row.get("session_date"))
    if not usable:
        return None

    stamp = computed_at or utc_now()
    grid = indicator_grid(usable)
    last = usable[-1]
    session = xcal.trading_session(session_date)
    event_at = session.rth_close_at if session else datetime(
        session_date.year, session_date.month, session_date.day, tzinfo=timezone.utc
    )

    features = {
        "symbol": symbol,
        "session_date": session_date,
        "feature_set_version": feature_set_version,
        "close": _number(last.get("close")),
        "atr14": atr(usable, ATR_LENGTH),
        "avwape_value": None,
        "avwape_upper_1": None,
        "avwape_upper_2": None,
        "avwape_upper_3": None,
        "avwape_lower_1": None,
        "avwape_lower_2": None,
        "avwape_lower_3": None,
        "favorite_zone_coord": None,
        "favorite_zone_residence_bars": None,
        "first_dev_touch_order": None,
        "band1_rejection_strength": None,
        "second_band_streak": None,
        # Was the anchor those bands come from knowable on this session (Q2.1)?
        # Set only where an anchor was actually used, so "" means unanchored
        # and NULL means the row predates the column.
        "anchor_knowledge": ANCHOR_KNOWLEDGE_UNANCHORED,
        # The CHALLENGER's bands (M4.1), from the SAME bars and the SAME anchor
        # index as the champion's. NULL until an anchor is used; NULL bands with
        # the formula version present mean "attempted and unmeasurable".
        "avwap_variant_value": None,
        "avwap_variant_stdev": None,
        "avwap_variant_upper_1": None,
        "avwap_variant_upper_2": None,
        "avwap_variant_upper_3": None,
        "avwap_variant_lower_1": None,
        "avwap_variant_lower_2": None,
        "avwap_variant_lower_3": None,
        "avwap_variant_formula_version": None,
        "ema8": _grid_value(grid, "ema_8"),
        "ema15": _grid_value(grid, "ema_15"),
        "ema21": _grid_value(grid, "ema_21"),
        "sma50": _grid_value(grid, "sma_50"),
        "sma100": _grid_value(grid, "sma_100"),
        "sma200": _grid_value(grid, "sma_200"),
        "dist_sma50_atr": None,
        "dist_sma100_atr": None,
        "dist_sma200_atr": None,
        "spy_regime_state": spy_regime_state,
        "input_manifest_hash": manifest_hash,
        "computed_at": stamp,
        "event_at": event_at,
        "input_capture_mode_worst": _worst_capture_mode(row.get("capture_mode") for row in usable),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }

    close = features["close"]
    atr_value = features["atr14"]
    if close is not None and atr_value:
        for period in (50, 100, 200):
            level = features[f"sma{period}"]
            if level is not None:
                features[f"dist_sma{period}_atr"] = (close - level) / atr_value

    if anchor_index is not None and 0 <= anchor_index < len(usable):
        # The CHALLENGER, computed FIRST and independently of whether the
        # champion produced anything (M4.1). The two formulas fail on different
        # inputs - the champion's sigma is zero on a one-bar anchor and the
        # challenger's is None until twenty closes exist - so gating one on the
        # other would silently drop a measured band. The formula version is
        # written whenever the challenger was ATTEMPTED, so a NULL band with a
        # version beside it reads as "not measurable here" rather than as a row
        # that predates the column.
        variant_vwap, variant_stdev, variant_bands = avwap_variant_bands(usable, anchor_index)
        features["avwap_variant_formula_version"] = _band_variants().FEATURE_VERSION
        features["avwap_variant_value"] = variant_vwap
        features["avwap_variant_stdev"] = variant_stdev
        for band in ("UPPER_1", "UPPER_2", "UPPER_3", "LOWER_1", "LOWER_2", "LOWER_3"):
            features[f"avwap_variant_{band.lower()}"] = variant_bands.get(band)

        avwap, _stdev, bands = anchored_vwap_bands(usable, anchor_index)
        if bands:
            features["anchor_knowledge"] = str(anchor_knowledge or ANCHOR_KNOWLEDGE_UNANCHORED)
            features["avwape_value"] = avwap
            for band in ("UPPER_1", "UPPER_2", "UPPER_3", "LOWER_1", "LOWER_2", "LOWER_3"):
                features[f"avwape_{band.lower()}"] = bands.get(band)
            zone = favorite_zone_block(usable[anchor_index:], avwap, bands)
            features["favorite_zone_coord"] = zone.coord
            features["favorite_zone_residence_bars"] = zone.residence_bars
            features["first_dev_touch_order"] = zone.first_dev_touch_order
            features["band1_rejection_strength"] = zone.band1_rejection_strength
            features["second_band_streak"] = zone.second_band_streak
    return features


def _grid_value(grid, column):
    if grid is None or column not in getattr(grid, "columns", []):
        return None
    value = grid[column].iloc[-1]
    return None if value is None or value != value else float(value)  # NaN-safe


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def anchor_index_for(d1_rows, anchor_bar_date: date) -> int | None:
    for index, row in enumerate(d1_rows):
        day = row.get("session_date")
        if isinstance(day, datetime):
            day = day.date()
        if day == anchor_bar_date:
            return index
    return None


def daily_history_window(
    store: ResearchStore,
    session_date: date,
    *,
    min_sessions: int = DAILY_HISTORY_MIN_SESSIONS,
    max_years: int = DAILY_HISTORY_MAX_YEARS,
):
    """The ``bar_d1`` partitions a tier-1 daily snapshot must read, and their rows.

    The stated ``tier1_v1`` rule: **always** read ``year`` and ``year-1`` - a
    200-session window spans roughly 9.5 calendar months, so a single year
    partition truncates the frame for every session from February onward - then
    keep walking back one year at a time until the deepest symbol in the frame
    holds ``min_sessions`` completed sessions on or before ``session_date``, or
    the lake runs out of years, or ``max_years`` partitions have been read.

    Returning the partitions alongside the rows keeps ``input_manifest_hash``
    honest: the hash covers exactly the files the snapshot was computed from.
    """
    partitions: set[tuple[str, str]] = set()
    rows_by_symbol: dict[str, list[dict]] = {}
    depth = 0

    for offset in range(int(max_years)):
        year = session_date.year - offset
        partition = ("bar_d1", f"year={year}")
        partitions.add(partition)
        found = 0
        for row in store.read_table(*partition).to_pylist():
            day = _as_date(row.get("session_date"))
            if day is None or day > session_date:
                continue
            rows_by_symbol.setdefault(str(row.get("symbol") or ""), []).append(row)
            found += 1
        depth = max((len(rows) for rows in rows_by_symbol.values()), default=0)
        # year and year-1 are the floor; past that, stop as soon as the window
        # is deep enough or the lake stops answering.
        if offset >= 1 and (depth >= int(min_sessions) or found == 0):
            break

    return partitions, rows_by_symbol


def build_daily_snapshots(
    store: ResearchStore | None,
    session_date: date,
    *,
    symbols=None,
    anchors_by_symbol=None,
    spy_regime_state: str | None = None,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "feature_snapshot_daily",
    feature_set_version: str = FEATURE_SET_VERSION,
) -> SnapshotReport:
    """Compute the D1 tier-1 snapshot for one session. Idempotent."""
    report = SnapshotReport(dataset="feature_snapshot_daily")
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    partitions, rows_by_symbol = daily_history_window(store, session_date)
    manifest_hash = input_manifest_hash(store, partitions)

    wanted = {str(symbol).strip().upper() for symbol in (symbols or [])} or set(rows_by_symbol)
    published = store.read_table(
        "feature_snapshot_daily",
        f"year={session_date.year}",
        columns=["symbol", "session_date", "feature_set_version"],
    ).to_pylist()
    existing = {
        (str(row["symbol"]), str(row["feature_set_version"]))
        for row in published
        if _as_date(row.get("session_date")) == session_date
    }

    rows = []
    for symbol in sorted(wanted):
        if (symbol, feature_set_version) in existing:
            report.skip("ALREADY_COMPUTED")
            continue
        d1_rows = rows_by_symbol.get(symbol) or []
        # A bare date still works (every caller before Q2.1 passed one); a
        # stamped AnchorChoice additionally says whether it was knowable.
        choice = _anchor_choice((anchors_by_symbol or {}).get(symbol))
        anchor_index = None
        anchor_knowledge = ANCHOR_KNOWLEDGE_UNANCHORED
        if choice is not None:
            ordered = sorted(
                [row for row in d1_rows if _as_date(row.get("session_date")) <= session_date],
                key=lambda row: row.get("session_date"),
            )
            anchor_index = anchor_index_for(ordered, choice.anchor_bar_date)
            if anchor_index is None:
                report.skip("ANCHOR_BAR_NOT_IN_HISTORY")
            else:
                anchor_knowledge = choice.knowledge
        computed = compute_daily_features(
            symbol,
            d1_rows,
            session_date=session_date,
            anchor_index=anchor_index,
            anchor_knowledge=anchor_knowledge,
            spy_regime_state=spy_regime_state,
            manifest_hash=manifest_hash,
            computed_at=stamp,
            feature_set_version=feature_set_version,
            run_id=run_id,
        )
        if computed is None:
            report.skip("NO_COMPLETED_BARS")
            continue
        rows.append(computed)

    if not rows:
        report.status = "NOTHING_TO_COMPUTE"
        return report
    report.rows = store.publish("feature_snapshot_daily", rows, job_id=job_id).rows_published
    report.symbols = len(rows)
    return report


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    return value


# ---------------------------------------------------------------------------
# Intraday snapshots
# ---------------------------------------------------------------------------
def compute_intraday_features(
    symbol: str,
    m5_rows,
    *,
    interval_start: datetime,
    session,
    derived_by_timeframe=None,
    prior_session=None,
    atr_value: float | None = None,
    context=None,
    computed_at: datetime | None = None,
    feature_set_version: str = FEATURE_SET_VERSION,
    run_id: str = "",
) -> dict | None:
    """One ``feature_snapshot_intraday`` row, keyed by an M5 ``interval_start``.

    The contract: the row keyed at bar S describes the state **through the
    close of bar S**, so exactly the bars up to and including S contribute and
    nothing later can leak in. Production context values arrive through
    ``context`` and are stored verbatim - this module never recomputes an RVOL,
    an RS/RW, or a chop-veto state.

    **Intraday EMA lookback rule** (``tier1_v1``): the M5 EMA frame is the
    *entry session's own RTH bars*, because that is the champion's frame -
    BounceBot fetches "5 D"/``useRTH=1`` for the previous-day extremes and the
    dynamic/EOD VWAPs, but computes ``ema_8/15/21`` on ``today_df`` alone and
    only once ``len(today_df) >= span`` (`bounce_bot_lib/legacy.py`, step 5,
    "Calculate short EMAs (today only)"). Seeding these EMAs on a multi-session
    frame would store a different number under the champion's column name -
    exactly the BD-33 failure the rule exists to prevent. The session bound is
    enforced here rather than trusted to the caller. M15/M30 have no champion
    (they are LD-23's new ground) and follow the same stated convention.
    """
    stamp = computed_at or utc_now()
    usable = [
        row
        for row in m5_rows
        if row.get("interval_start") is not None
        and row["interval_start"] <= interval_start
        and session.rth_open_at <= row["interval_start"] < session.rth_close_at
        and row.get("is_complete", True)
    ]
    usable.sort(key=lambda row: row["interval_start"])
    if not usable:
        return None
    closes = [_number(row.get("close")) for row in usable]
    vwap, upper, lower = session_vwap_bands(usable)
    supplied = dict(context or {})

    row = {
        "symbol": symbol,
        "interval_start": interval_start,
        "session_id": session.session_id,
        "session_phase": session.phase_of(interval_start),
        "feature_set_version": feature_set_version,
        "session_vwap": _number(vwap),
        "session_vwap_upper_1": _number(upper),
        "session_vwap_lower_1": _number(lower),
        "vwap_algorithm": VWAP_ALGORITHM_STANDARD,
        "ema8_m5": ema_series(closes, 8, min_bars=8),
        "ema15_m5": ema_series(closes, 15, min_bars=15),
        "ema21_m5": ema_series(closes, 21, min_bars=21),
        "ema8_m15": None,
        "ema15_m15": None,
        "ema21_m15": None,
        "ema8_m30": None,
        "ema15_m30": None,
        "ema21_m30": None,
        # Production context: recorded as production computed it, or null.
        "rvol_tc2000": supplied.get("rvol_tc2000"),
        "rvol_gate_pass": supplied.get("rvol_gate_pass"),
        "rs_rw_vs_spy": supplied.get("rs_rw_vs_spy"),
        "group_rs_debiased": supplied.get("group_rs_debiased"),
        "market_internals_negative": supplied.get("market_internals_negative"),
        "session_structure_gate": supplied.get("session_structure_gate"),
        "pullback_count_in_current_leg": supplied.get("pullback_count_in_current_leg"),
        "dist_pdh_atr": None,
        "dist_pdl_atr": None,
        "computed_at": stamp,
        "observed_at": supplied.get("observed_at") or stamp,
        "capture_mode": _worst_capture_mode(bar.get("capture_mode") for bar in usable),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }

    for timeframe, prefix in (("M15", "m15"), ("M30", "m30")):
        derived = [
            bar
            for bar in (derived_by_timeframe or {}).get(timeframe, [])
            if bar.get("interval_end") is not None and bar["interval_end"] <= interval_start
        ]
        derived.sort(key=lambda bar: bar["interval_start"])
        if derived:
            values = [_number(bar.get("close")) for bar in derived]
            for span in (8, 15, 21):
                row[f"ema{span}_{prefix}"] = ema_series(values, span, min_bars=span)

    last_close = closes[-1] if closes else None
    if prior_session and atr_value and last_close is not None:
        pdh = _number(prior_session.get("high"))
        pdl = _number(prior_session.get("low"))
        if pdh is not None:
            row["dist_pdh_atr"] = (last_close - pdh) / atr_value
        if pdl is not None:
            row["dist_pdl_atr"] = (last_close - pdl) / atr_value
    return row


def build_intraday_snapshots(
    store: ResearchStore | None,
    session_date: date,
    *,
    symbols=None,
    boundaries=None,
    context_by_symbol=None,
    prior_session_by_symbol=None,
    atr_by_symbol=None,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "feature_snapshot_intraday",
    feature_set_version: str = FEATURE_SET_VERSION,
) -> SnapshotReport:
    """Cohort intraday snapshots at completed M5 boundaries."""
    report = SnapshotReport(dataset="feature_snapshot_intraday")
    if store is None:
        report.status = "DISABLED"
        return report
    session = xcal.trading_session(session_date)
    if session is None:
        report.status = "NOTHING_TO_COMPUTE"
        return report
    stamp = now or utc_now()
    partition = f"month={session.rth_open_at:%Y-%m}"

    # The session window runs in ARROW, not in Python: this used to read the
    # whole MONTH partition into dicts (8.7M rows / 15.4 GB on 2026-08-27) to
    # keep one session of it. The predicate is the same half-open window.
    #
    # The SYMBOL filter is applied only when the caller named symbols. With
    # none named, `wanted` below is derived from the bars themselves - the
    # cohort present in this session - so narrowing the read would change the
    # answer rather than just its cost.
    named = sorted({str(symbol).strip().upper() for symbol in (symbols or [])})
    window = (session.rth_open_at, session.rth_close_at)

    m5_by_symbol: dict[str, list[dict]] = {}
    for row in store.read_rows(
        "bar_m5", partition, symbols=named or None, interval_start_range=window
    ):
        if row.get("interval_start") is None:
            continue
        m5_by_symbol.setdefault(str(row.get("symbol") or ""), []).append(row)

    derived_by_symbol: dict[str, dict[str, list[dict]]] = {}
    for timeframe in ("M15", "M30"):
        for row in store.read_rows(
            "bar_derived",
            f"timeframe={timeframe}/month={session.rth_open_at:%Y-%m}",
            symbols=named or None,
            interval_start_range=window,
        ):
            if row.get("interval_start") is None:
                continue
            derived_by_symbol.setdefault(str(row.get("symbol") or ""), {}).setdefault(timeframe, []).append(row)

    wanted = set(named) or set(m5_by_symbol)
    existing = {
        (str(row["symbol"]), row["interval_start"])
        for row in store.read_table(
            "feature_snapshot_intraday", partition, columns=["symbol", "interval_start"]
        ).to_pylist()
    }

    rows = []
    for symbol in sorted(wanted):
        bars = m5_by_symbol.get(symbol) or []
        if not bars:
            report.skip("NO_BARS")
            continue
        # Rows are keyed by completed M5 interval starts (the dataset grain).
        points = boundaries or sorted({bar["interval_start"] for bar in bars})
        for boundary in points:
            if (symbol, boundary) in existing:
                report.skip("ALREADY_COMPUTED")
                continue
            computed = compute_intraday_features(
                symbol,
                bars,
                interval_start=boundary,
                session=session,
                derived_by_timeframe=derived_by_symbol.get(symbol),
                prior_session=(prior_session_by_symbol or {}).get(symbol),
                atr_value=(atr_by_symbol or {}).get(symbol),
                context=(context_by_symbol or {}).get(symbol),
                computed_at=stamp,
                feature_set_version=feature_set_version,
                run_id=run_id,
            )
            if computed is not None:
                rows.append(computed)

    if not rows:
        report.status = "NOTHING_TO_COMPUTE"
        return report
    report.rows = store.publish("feature_snapshot_intraday", rows, job_id=job_id).rows_published
    report.symbols = len({row["symbol"] for row in rows})
    return report


__all__ = [
    "ATR_LENGTH",
    "AVWAP_FORMULA_VERSION",
    "DAILY_HISTORY_MAX_YEARS",
    "DAILY_HISTORY_MIN_SESSIONS",
    "FEATURE_SET_VERSION",
    "FavoriteZone",
    "SnapshotReport",
    "VWAP_ALGORITHM_STANDARD",
    "anchor_index_for",
    "anchored_vwap_bands",
    "atr",
    "avwap_variant_bands",
    "build_anchor_instances",
    "build_daily_snapshots",
    "build_intraday_snapshots",
    "compute_daily_features",
    "compute_intraday_features",
    "daily_history_window",
    "ema_series",
    "favorite_zone_block",
    "indicator_grid",
    "input_manifest_hash",
    "session_vwap_bands",
]
