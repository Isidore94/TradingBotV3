"""The `swing_path_facts` night slot (S15 items 3 and 9): deterministic, no model.

Reads the session-horizon outcomes, `d1_features_history.csv` and the daily bars,
builds `swing_path_facts.build_path_fact_rows` (MFE/MAE in ATR over horizons
1/3/5/10/20, next-open and leader-pullback fills) and replaces
`SWING_PATH_FACTS_FILE`. No rows writes nothing; a failed read, build or write
returns `failed` and the last good file stays. This slot is the file's one owner.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _last_completed(session_date: str) -> date:
    text = str(session_date or "")[:10]
    if text:
        return date.fromisoformat(text)
    import market_calendar

    return market_calendar.last_completed_session(datetime.now())


def run_swing_path_facts(
    *,
    session_date: str = "",
    horizon_path: Any = None,
    features_path: Any = None,
    bars_dir: Any = None,
    out_path: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import project_paths as pp
    import swing_path_facts as spf

    horizon_path = Path(horizon_path or pp.MASTER_AVWAP_SESSION_HORIZON_OUTCOMES_FILE)
    features_path = Path(features_path or pp.D1_FEATURES_HISTORY_FILE)
    bars_dir = Path(bars_dir or pp.MASTER_AVWAP_DAILY_BARS_DIR)
    out_path = Path(out_path or pp.SWING_PATH_FACTS_FILE)
    try:
        with horizon_path.open("r", newline="", encoding="utf-8-sig") as handle:
            horizon_rows = list(csv.DictReader(handle))
        wanted = {str(row.get("scan_row_id") or "").strip() for row in horizon_rows}
        facts = spf.scan_facts_from_features(features_path, wanted)
        build = spf.build_path_fact_rows(
            horizon_rows, spf.daily_bars_from_dir(bars_dir), facts,
            last_completed_session=_last_completed(session_date),
        )
    except Exception as exc:  # noqa: BLE001 - the night goes on; the last file stays
        _log.exception("swing_path_facts: build failed")
        return {"status": "failed", "model": "",
                "reason": f"swing path facts not built ({type(exc).__name__}: {exc}); last file kept",
                "outputs": []}
    if not build.rows:
        return {"status": "ok", "model": "",
                "reason": "no session-horizon rows to measure; nothing written, last file kept",
                "outputs": []}
    try:
        written = spf.write_rows(build.rows, out_path)
    except OSError as exc:
        return {"status": "failed", "model": "",
                "reason": f"swing path facts not written ({exc}); last file kept", "outputs": []}
    measured = sum(1 for row in build.rows if row.get("measured") is True)
    return {"status": "ok", "model": "",
            "reason": (f"{build.entries} scan rows -> {len(build.rows)} path rows "
                       f"({measured} measured, {build.scan_facts_missing} with no scan facts)"),
            "outputs": [str(written)]}
