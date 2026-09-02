"""The nightly evidence report — R10.I, under the recorded sequencing override.

R10.I was specified to be built "after two weeks of R10.A collection". The
trader waived that **sequencing** on 2026-08-24 and explicitly did not waive the
gate it exists to protect (decision record §4):

> **NOT waived: the evidence-quality gate on claims.** Until two weeks of
> R10.A/B collection exist, every report the slot emits must state its n, label
> everything `discovery`, and say in words that the collection window is not
> met. A report over a near-empty ledger is honest scaffolding, never a finding.

So this runs now and is scaffolding now, and the scaffolding says so on its own
first line. The collection clock starts at the first live session after the
trader flipped `outcome_sweep_autorun="on"` (2026-08-24); until it has run its
course, every number here is a shape rather than a result.

**Deterministic. No model is called**, nothing is sent anywhere, and nothing in
this chain may reach a detector, score, alert, watchlist, Focus, the review
queue or `review_policy.json` — a test walks this module's AST to keep that
true rather than trusting the sentence.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)

#: Schema NAME (ground rule 5).
REPORT_SCHEMA = "evidence_report_v1"

#: The window R10.A/B collection must clear before a number here is a finding.
#: Sessions, not calendar days - two weeks of TRADING is the unit the gate names.
REQUIRED_COLLECTION_SESSIONS = 10

#: Printed verbatim at the top of every report until the window is met. The
#: precise wording matters: a reader who skims must not be able to mistake this
#: for a hedge on a real finding.
UNMET_WINDOW_STATEMENT = (
    "COLLECTION WINDOW NOT MET. This report is honest scaffolding, not a "
    "finding. R10.I's evidence-quality gate requires two weeks of R10.A/B "
    "collection before any number here may be read as evidence; nothing below "
    "has cleared it. Every figure is labelled `discovery` and carries its n. "
    "Do not promote, demote, or change anything on the strength of this file."
)

MET_WINDOW_STATEMENT = (
    "Collection window met. Every figure is still labelled `discovery` unless "
    "it names a window declared in advance - a large post-hoc sample is a large "
    "discovery, never a confirmation."
)


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def collection_state(sessions_collected: int) -> dict[str, Any]:
    """How far the R10.A/B collection window has run.

    `sessions_collected` is COUNTED by the caller from the ledger, never
    estimated here. A caller that cannot count returns 0, which reads as
    "unmet" - the conservative direction, and the only one that cannot turn
    scaffolding into a finding by accident.
    """
    collected = max(0, int(sessions_collected or 0))
    met = collected >= REQUIRED_COLLECTION_SESSIONS
    return {
        "sessions_collected": collected,
        "sessions_required": REQUIRED_COLLECTION_SESSIONS,
        "window_met": met,
        "statement": MET_WINDOW_STATEMENT if met else UNMET_WINDOW_STATEMENT,
    }


def build_report(
    *,
    session_date: str,
    ledger_rows: int = 0,
    sessions_collected: int = 0,
    cohorts: Mapping[str, Any] | None = None,
    scoreboard: Mapping[str, Any] | None = None,
    unavailable: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One night's deterministic evidence report.

    Every section carries its n and its `discovery` label. Nothing here
    computes a statistic: the numbers arrive already measured, through
    `evidence_stats`, from the surfaces that own them.
    """
    import evidence_stats

    state = collection_state(sessions_collected)
    missing = {str(name): str(reason) for name, reason in (unavailable or {}).items()}
    sections = {
        "cohorts": dict(cohorts or {}),
        "scoreboard": dict(scoreboard or {}),
    }
    counted = {
        name: _row_count(payload) for name, payload in sections.items()
    }
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": _now(now).isoformat(timespec="seconds"),
        "session_date": str(session_date or ""),
        # First key, first line rendered, and the one a skimming reader sees.
        "window": state,
        "evidence_label": evidence_stats.LABEL_DISCOVERY,
        "statistics_contract": {
            "module": "evidence_stats",
            "schema": evidence_stats.SUMMARY_SCHEMA,
            "n_floor": evidence_stats.MIN_REPORTABLE_N,
            "n_floor_note": "necessary, never sufficient",
        },
        "ledger_rows": int(ledger_rows or 0),
        "counts": counted,
        "sections": sections,
        "unavailable": missing,
        "summary": _summary(state, counted, int(ledger_rows or 0), missing),
    }


def _row_count(payload: Any) -> int:
    """Rows in a section, counting one level of nesting.

    The cohorts section is `{"veto": [...], "like": [...]}`, so counting only
    the top level reported 0 while both cohorts held rows - an n of zero beside
    real data is worse than no n at all, because it reads as "nothing measured".
    """
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return len(rows)
        return sum(len(value) for value in payload.values() if isinstance(value, list))
    return 0


def _summary(
    state: Mapping[str, Any], counts: Mapping[str, int], ledger_rows: int, missing: Mapping[str, str]
) -> str:
    parts = [str(state.get("statement") or "")]
    parts.append(
        f"n: {ledger_rows} outcome ledger row(s); "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
        + f". Collection: {state.get('sessions_collected')} of "
        f"{state.get('sessions_required')} session(s)."
    )
    if missing:
        named = ", ".join(f"{name} ({reason})" for name, reason in sorted(missing.items()))
        parts.append(
            f"{len(missing)} source(s) could not be read, so this report is "
            f"INCOMPLETE rather than empty: {named}."
        )
    return " ".join(parts)


def render_markdown(report: Mapping[str, Any]) -> str:
    """The human-readable half. The window statement is the first thing on it."""
    lines = [f"# Evidence report - {report.get('session_date', '')}\n\n"]
    lines.append(f"**{(report.get('window') or {}).get('statement', '')}**\n\n")
    lines.append(f"{report.get('summary', '')}\n\n")
    lines.append(
        f"Every figure below is labelled `{report.get('evidence_label')}`. "
        f"Statistics come from `{(report.get('statistics_contract') or {}).get('module')}`; "
        f"n >= {(report.get('statistics_contract') or {}).get('n_floor')} is "
        "necessary, never sufficient.\n\n"
    )
    for name, payload in sorted((report.get("sections") or {}).items()):
        count = (report.get("counts") or {}).get(name, 0)
        lines.append(f"## {name} (n={count}, discovery)\n\n")
        if not count:
            lines.append("Nothing measured for this section.\n\n")
            continue
        lines.append("```json\n" + json.dumps(payload, indent=1, sort_keys=True)[:4000] + "\n```\n\n")
    return "".join(lines)


def run_evidence_report(
    *,
    session_date: str = "",
    now: datetime | None = None,
    report_dir: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The `evidence_report` runner slot. Deterministic; calls no model.

    Reads what the evidence surfaces already produced and writes one report
    with atomic last-good. A source it cannot read is NAMED in the report
    rather than silently omitted, because a report that is empty because a file
    would not open must not read as a quiet night.
    """
    from ai_jobs import ledger as job_ledger

    unavailable: dict[str, str] = {}
    ledger_rows = 0
    sessions_collected = 0
    cohorts: dict[str, Any] = {}
    scoreboard: dict[str, Any] = {}

    try:
        from evidence_ledger import intraday_outcome_ledger

        result = intraday_outcome_ledger().read()
        ledger_rows = len(result.rows)
        sessions_collected = len(
            {str(row.get("session_date") or "") for row in result.rows if row.get("session_date")}
        )
    except Exception as exc:  # noqa: BLE001
        unavailable["intraday outcome ledger"] = str(exc)

    for name, loader in (
        ("veto", _read_veto_cohort_rows),
        ("like", _read_like_cohort_rows),
        # P5, APPENDED. With these the report covers every verdict the trader
        # can record rather than only the two that had graders first.
        ("pass", _read_pass_cohort_rows),
        ("rejection", _read_rejection_cohort_rows),
    ):
        try:
            cohorts[name] = loader()
        except Exception as exc:  # noqa: BLE001
            unavailable[f"{name} cohort"] = str(exc)

    try:
        scoreboard = _read_scoreboard_bundle()
    except Exception as exc:  # noqa: BLE001
        unavailable["setup scoreboard bundle"] = str(exc)

    report = build_report(
        session_date=session_date or date.today().isoformat(),
        ledger_rows=ledger_rows,
        sessions_collected=sessions_collected,
        cohorts=cohorts,
        scoreboard=scoreboard,
        unavailable=unavailable,
        now=now,
    )

    outputs: list[str] = []
    try:
        target_dir = Path(report_dir) if report_dir is not None else _default_report_dir()
        outputs.append(str(_publish(target_dir / "evidence_report.json",
                                    json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")))
        outputs.append(str(_publish(target_dir / "evidence_report.md", render_markdown(report))))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": job_ledger.STATUS_FAILED,
            "model": "",
            "reason": f"evidence report could not be published: {exc}",
            "outputs": [],
        }

    window = report["window"]
    return {
        "status": job_ledger.STATUS_OK,
        "model": "",
        "reason": (
            ("window NOT met - scaffolding only; " if not window["window_met"] else "")
            + f"n={ledger_rows} ledger row(s) over {window['sessions_collected']} session(s)"
            + (f"; {len(unavailable)} source(s) unreadable" if unavailable else "")
        ),
        "outputs": outputs,
    }


def _default_report_dir() -> Path:
    from setup_scoreboard import report_store_dir

    return Path(report_store_dir())


def _publish(path: Path, content: str) -> Path:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _read_veto_cohort_rows() -> list[dict[str, str]]:
    import csv

    from project_paths import VETO_COHORT_PERFORMANCE_FILE

    path = Path(VETO_COHORT_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_like_cohort_rows() -> list[dict[str, str]]:
    import csv

    from project_paths import LIKE_COHORT_PERFORMANCE_FILE

    path = Path(LIKE_COHORT_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_pass_cohort_rows() -> list[dict[str, str]]:
    """The day-trade PASS rollup.

    Its code cohorts OVERLAP by construction - a pass with k codes is in k of
    them plus `pass_all` - so a reader must never sum them. The fact pack
    carries that sentence beside the rows.
    """
    import csv

    from project_paths import PASS_COHORT_PERFORMANCE_FILE

    path = Path(PASS_COHORT_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_rejection_cohort_rows() -> list[dict[str, str]]:
    """The NOT-TODAY and DISLIKE rollup. Separate cohorts, never pooled."""
    import csv

    from project_paths import REJECTION_COHORT_PERFORMANCE_FILE

    path = Path(REJECTION_COHORT_PERFORMANCE_FILE)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_scoreboard_bundle() -> dict[str, Any]:
    from setup_scoreboard import report_store_dir

    path = Path(report_store_dir()) / "setup_scoreboard.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
