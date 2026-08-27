#!/usr/bin/env python3
"""Freeze the `avwap_band_variant_oneoption_v1` golden fixture (Phase 0.10 B-0).

The bars are OKTA, 2026-04-01 through 2026-06-05, taken from the durable daily
store on the desk and passed through **the champion's own normalisation path**
(`master_avwap_lib.legacy._normalize_daily_bar_frame`) before they are written
here. That detail is the whole reason this script exists rather than a hand
copy: the store's OKTA volumes are mixed-unit - thousands before 2026-05-27 and
again on 2026-06-04, shares otherwise, with `volume_unit = unknown` on every row
- and the study (`docs/AVWAP_BAND_VARIANT_STUDY.md` section 2b) recorded that
hazard with the instruction that the real fit must go through the champion's
normalisation and never an ad-hoc threshold. Freezing the normalised frame is
how that instruction is kept: the fixture is what the champion would have been
handed, splice and all.

The splice does not reach either golden row, and that is a measurement rather
than a hope: the anchored centre only sums bars from 2026-05-29 forward, and the
first mixed-unit bar after the anchor is 2026-06-04, two sessions after the last
golden date. The sigma uses no volume at all.

The expectations are the trader's hover readings off OneOption / Option Stalker
Pro itself, not this repository's output - a golden value has to come from the
thing being replicated or it proves only that the code agrees with itself.

Run:  .venv/Scripts/python.exe scripts/build_avwap_band_variant_fixture.py
It refuses to overwrite an existing fixture without --force: a golden file that
can be silently regenerated is not golden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "avwap_band_variant_oneoption_v1.json"

SYMBOL = "OKTA"
FIRST_DATE = "2026-04-01"
LAST_DATE = "2026-06-05"
ANCHOR_DATE = "2026-05-29"
LOOKBACK = 20

# The trader's hover readings off OneOption, 2026-08-26 evening, recorded in
# docs/AVWAP_BAND_VARIANT_STUDY.md section 2b. `centre` on 2026-06-02 is the
# midpoint of the two band readings; the anchor bar's centre was read directly.
VENDOR_READINGS = [
    {
        "date": "2026-05-29",
        "centre": 118.19,
        "upper_1": 128.47,
        "lower_1": 107.90,
        "note": (
            "One-bar anchor. The centre is exactly that bar's HLC/3 (118.187), which is "
            "how the study killed OHLC/4 (115.52) and every anchor-offset candidate: a "
            "VWAP equal to one bar's typical price to the cent cannot contain an earlier "
            "bar. sigma = 10.28 here, where the champion's is 0.0 by construction."
        ),
    },
    {
        "date": "2026-06-02",
        "centre": 126.565,
        "upper_1": 144.60,
        "lower_1": 108.53,
        "note": (
            "Two hovers, upper then lower; the centre is their midpoint and the "
            "half-width is 18.035. This is the reading that killed the sample-OHLC "
            "form, which predicts 138.09 here. Our centre from the store's volumes is "
            "126.78, a 0.17% gap that is a consolidated-vs-IB volume-feed difference "
            "and not a formula difference - hence the 0.2% centre tolerance."
        ),
    },
]

# Asserted separately from the numeric_tolerance, which the contract applies
# structurally: the centre is allowed a relative 0.2% (a volume-feed gap) while
# the sigma, which uses no volume, must reproduce to +/-0.02 absolute.
CENTRE_RELATIVE_TOLERANCE = 0.002
SIGMA_ABSOLUTE_TOLERANCE = 0.02


def _frame():
    import pandas as pd

    from master_avwap_lib.legacy import _normalize_daily_bar_frame
    from project_paths import MASTER_AVWAP_DAILY_BARS_DIR

    path = Path(MASTER_AVWAP_DAILY_BARS_DIR) / f"{SYMBOL}.parquet"
    if not path.exists():
        raise SystemExit(f"durable daily store has no {SYMBOL}: {path}")
    frame = _normalize_daily_bar_frame(pd.read_parquet(path)).reset_index(drop=True)
    window = frame[
        (frame["datetime"] >= FIRST_DATE) & (frame["datetime"] <= LAST_DATE)
    ].reset_index(drop=True)
    if window.empty:
        raise SystemExit(f"{SYMBOL}: no bars between {FIRST_DATE} and {LAST_DATE}")
    return window


def _bars(frame) -> list[dict]:
    out: list[dict] = []
    for _, row in frame.iterrows():
        volume = row["volume"]
        out.append(
            {
                "date": row["datetime"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                # A row may legitimately carry no volume (R10.V step 3); keep the
                # absence rather than inventing a zero, which the band function
                # would treat identically but a reader would not.
                "volume": None if volume != volume else float(volume),
            }
        )
    return out


def build() -> dict:
    bars = _bars(_frame())
    dates = [bar["date"] for bar in bars]
    if ANCHOR_DATE not in dates:
        raise SystemExit(f"anchor {ANCHOR_DATE} is not a session in the frozen window")
    for reading in VENDOR_READINGS:
        if reading["date"] not in dates:
            raise SystemExit(f"reading date {reading['date']} is not in the frozen window")

    payload = {
        # --- repo-wide fixture contract (plan.md Milestone 3) ----------------
        "schema": "avwap_band_variant_oneoption_v1",
        "feature_version": "avwap_bands_oneoption_bb20_v1",
        "raw_input_keys": ["bars", "rules_under_test"],
        "raw_input_sha256": "",
        "acquired_at": "2026-08-26T21:30:00-04:00",
        "universe_version": "durable_daily_bars_2026-08-26",
        "provider_assumptions": (
            "OKTA daily bars read once from the desk's durable store "
            "(data/daily_bars/OKTA.parquet) and passed through the champion's own "
            "_normalize_daily_bar_frame before freezing, never through an ad-hoc "
            "volume threshold. The store's OKTA volumes are mixed-unit - thousands "
            "before 2026-05-27 and again on 2026-06-04, shares otherwise, with "
            "volume_unit='unknown' throughout - and the frozen bars carry that splice "
            "exactly as the champion would receive it. Neither golden row is affected: "
            "the anchored centre sums only 2026-05-29 forward and the first mixed-unit "
            "bar after the anchor is 2026-06-04, while the sigma uses no volume at all. "
            "The expected values are the trader's hover readings off OneOption / Option "
            "Stalker Pro, not this repository's output."
        ),
        "as_of": "2026-06-05T16:00:00-04:00",
        "expected_keys": ["expected"],
        # Structural default. The two assertions that matter carry their own,
        # tighter and looser respectively: sigma to +/-0.02 absolute, centre to
        # 0.2% relative. Both live in rules_under_test so the test reads them
        # from the fixture rather than restating them.
        "numeric_tolerance": 0.02,
        "intentional_difference": (
            "The centre is expected to differ from the vendor by up to 0.2%. OneOption "
            "prices its AVWAP on consolidated volume and this store carries IB volume, "
            "so the weights differ; on 2026-06-02 that is 126.78 here against 126.565 "
            "there. The anchor bar matches to the cent because a single bar's VWAP does "
            "not depend on volume at all. The sigma needs no volume and reproduces from "
            "any clean close series, so it is held to +/-0.02."
        ),
        # --- the fixture itself ----------------------------------------------
        "fixture": "avwap_band_variant_oneoption_v1",
        "frozen_at": "2026-08-26",
        "symbol": SYMBOL,
        "note": (
            "Phase 0.10 B-0. Replication target for "
            "scripts/indicators/avwap_band_variants.py. The killed sample-OHLC form - "
            "sample stdev (n-1) of every O/H/L/C print since the anchor around the "
            "running AVWAP - predicts an upper band of 138.09 on 2026-06-02 against the "
            "144.60 the trader read, and is asserted as a discriminator in "
            "tests/test_avwap_band_variants.py."
        ),
        "rules_under_test": {
            "anchor_date": ANCHOR_DATE,
            "lookback": LOOKBACK,
            "ddof": 0,
            "typical_price": "hlc3",
            "centre": "anchored volume-weighted VWAP of HLC/3 from the anchor bar to t",
            "sigma": "population standard deviation of the last 20 closes ending at t",
            "sigma_window_crosses_the_anchor": True,
            "zero_or_nan_volume_bars": "skipped in the centre; their closes still count in sigma",
            "centre_relative_tolerance": CENTRE_RELATIVE_TOLERANCE,
            "sigma_absolute_tolerance": SIGMA_ABSOLUTE_TOLERANCE,
        },
        "bars": bars,
        "expected": {
            "vendor": "OneOption / Option Stalker Pro",
            "readings": VENDOR_READINGS,
        },
    }
    payload["raw_input_sha256"] = hashlib.sha256(
        json.dumps(
            {key: payload[key] for key in payload["raw_input_keys"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the OneOption AVWAP band golden fixture")
    parser.add_argument("--force", action="store_true", help="overwrite an existing fixture")
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        print(f"refusing to overwrite {args.out} (pass --force if you mean it)")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
