"""Deterministic, novice-friendly explanations for research and tracker rows.

These explanations translate existing evidence; they never invent an entry or
promote a research association into a live signal.  Keeping the translation
pure also makes it contract-testable and safe to reuse in exports/A.I. scopes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from setup_docs import resolve_setup_doc


def _text(value: Any, default: str = "") -> str:
    result = str(value or "").strip()
    return result or default


def _float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return 0 if value in (None, "") else int(float(value))
    except (TypeError, ValueError):
        return 0


def _signed(value: Any, suffix: str = "") -> str:
    numeric = _float(value)
    return "n/a" if numeric is None else f"{numeric:+.2f}{suffix}"


def _percent(value: Any) -> str:
    numeric = _float(value)
    if numeric is None:
        return "n/a"
    if abs(numeric) <= 1.0:
        numeric *= 100.0
    return f"{numeric:.0f}%"


def _sample_caution(sample_count: int, *, research_only: bool = False) -> str:
    if research_only:
        return (
            "This is research context, not an entry signal. Wait for a qualified setup and its own "
            "completed-bar trigger, stop, and target plan."
        )
    if sample_count <= 0:
        return "No usable sample count is attached, so treat the row as descriptive only."
    if sample_count < 12:
        return f"Only {sample_count} samples are measured. That is early evidence, not a dependable edge."
    return f"This summarizes {sample_count} observations; it is evidence, not a guarantee on the next trade."


def _setup_explanation(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    family = _text(row.get("setup_family"), "General setup")
    doc_key, doc = resolve_setup_doc(family)
    side = _text(row.get("side"), "LONG").upper()
    symbol = _text(row.get("symbol")).upper()
    label = _text(doc.get("label"), family)
    sample_count = max(
        _int(row.get("closed_setups")),
        _int(row.get("samples_2d")),
        _int(row.get("observation_count")),
    )

    steps = [
        f"Set the direction first: this row is for {side}. A short uses the same logic upside-down.",
        f"Wait for the entry confirmation: {_text(doc.get('entry'), 'use the setup alert’s completed-bar trigger.')}",
        f"Define the invalidation before entering: {_text(doc.get('stop'), 'use the protective level named by the setup.')}",
        f"Manage the winner using the measured plan: {_text(doc.get('targets'), 'take partial profit, then trail the remainder.')}",
    ]
    if kind == "setup_playbook" and row.get("stop_reference_label"):
        steps[2] = (
            f"For this measured playbook variant, the stop reference is {_text(row.get('stop_reference_label'))}. "
            "The row's exit plan is " + _text(row.get("profit_take_summary"), "not available") + "."
        )

    evidence: list[str] = []
    if row.get("avg_r_2d") not in (None, ""):
        evidence.append(
            f"After two sessions: average {_signed(row.get('avg_r_2d'), 'R')}, median "
            f"{_signed(row.get('median_r_2d'), 'R')}, win rate {_percent(row.get('win_rate_2d'))} "
            f"across {_int(row.get('samples_2d'))} samples."
        )
    if row.get("avg_closed_r") not in (None, ""):
        evidence.append(
            f"Closed outcomes: average {_signed(row.get('avg_closed_r'), 'R')}, target hit "
            f"{_percent(row.get('target_hit_rate'))}, stop rate {_percent(row.get('stop_rate'))} "
            f"across {_int(row.get('closed_setups'))} closed setups."
        )
    if row.get("robust_closed_r") not in (None, ""):
        evidence.append(
            f"This stop/exit combination measured {_signed(row.get('robust_closed_r'), 'R')} robust closed R "
            f"across {_int(row.get('closed_setups'))} closed setups."
        )
    if row.get("priority_score") not in (None, ""):
        evidence.append(
            f"The current scanner classification is tier {_text(row.get('tier'), '?')} with priority score "
            f"{_text(row.get('priority_score'), 'n/a')}. That rank chooses attention; it does not replace the trigger."
        )
    if not evidence:
        evidence.append(_text(doc.get("evidence"), "No measured outcome summary is attached to this row."))

    subject = f"{symbol} {side}" if symbol else f"{side} {label}"
    return {
        "title": subject,
        "eyebrow": "How this setup is executed",
        "summary": _text(doc.get("what"), f"A measured {label} setup."),
        "steps": steps,
        "evidence": evidence,
        "caution": _sample_caution(sample_count),
        "setup_doc_key": doc_key,
    }


def _daytrade_explanation(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    side = _text(row.get("direction"), "both sides").upper()
    dimension = _text(row.get("dimension"), "segment").replace("_", " ")
    segment = _text(row.get("segment"), "unknown")
    samples = _int(row.get("sample_count"))
    status = _text(row.get("status") or row.get("recommendation"), "measured")
    evidence = [
        f"Measured result: average {_signed(row.get('avg_close_r'), 'R')}, median "
        f"{_signed(row.get('median_close_r'), 'R')}, 1R hit {_percent(row.get('target_1r_rate'))}, "
        f"stop rate {_percent(row.get('stop_rate'))}, n={samples}."
    ]
    if row.get("score_delta") not in (None, ""):
        evidence.append(
            f"Live learning adjustment: {_signed(row.get('score_delta'))} score points; status {status}. "
            "This modifies alert emphasis only within the existing learning rules."
        )
    return {
        "title": f"{side} · {dimension}: {segment}",
        "eyebrow": "How to use this Day Trade Tracker row",
        "summary": (
            "This row grades a situation in which BounceBot alerts occurred. It is not a price level and cannot "
            "be traded by itself."
        ),
        "steps": [
            f"First wait until the bot identifies a {side} candidate whose {dimension} is '{segment}'.",
            "Treat an approaching alert as a preview. Wait for the completed M5 candle and the confirmed bounce alert.",
            "The confirmed alert names the level that held. Use that level to define invalidation before deciding size; 1R is the amount lost if invalidated.",
            "Only then use this row as a confidence check. Manage the exit from the actual alert/setup plan, not from this aggregate row.",
        ],
        "evidence": evidence,
        "caution": _sample_caution(samples),
    }


def _forensics_explanation(row: Mapping[str, Any]) -> dict[str, Any]:
    side = _text(row.get("side"), "BOTH").upper()
    pattern = _text(row.get("pattern"), "Unnamed condition combination")
    lift = _float(row.get("lift"))
    lift_text = "n/a" if lift is None else f"{lift:.2f}x"
    return {
        "title": f"{side} move pattern",
        "eyebrow": "How to use this Move Forensics result",
        "summary": (
            f"The historical condition combination '{pattern}' appeared before large {side.lower()} moves more "
            "or less often than it appeared on ordinary comparison days."
        ),
        "steps": [
            "Do not enter because this pattern is present. It was discovered by looking backward from outcomes.",
            "Use the row to raise or lower attention on a stock, then require a separate qualified setup and completed-bar trigger.",
            "Take the stop and targets from that real setup. Move Forensics supplies context, not executable prices.",
            "Validate a novel pattern on later, unseen sessions before proposing any scanner or score change.",
        ],
        "evidence": [
            f"Mover rate {_percent(row.get('mover_rate'))} versus ordinary baseline "
            f"{_percent(row.get('baseline_rate'))}; lift {lift_text}; seen before "
            f"{_int(row.get('movers_with'))} qualifying moves; average move "
            f"{_signed(row.get('avg_move_atr'), ' ATR')}.",
            "'Lift' means relative frequency, not expected profit and not probability that the next stock will move.",
        ],
        "caution": _sample_caution(_int(row.get("movers_with")), research_only=True),
    }


def _generic_research_explanation(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    labels = {
        "setup_scan_factor": "Scan factor",
        "setup_tier_performance": "Tier performance",
        "setup_catch_rate": "Catch-rate audit",
        "setup_human_pick": "Human-vs-bot comparison",
    }
    label = labels.get(kind, "Research row")
    samples = max(_int(row.get("observation_count")), _int(row.get("opportunity_count")), _int(row.get("pick_count")))
    evidence = ", ".join(
        f"{str(key).replace('_', ' ')}={value}"
        for key, value in row.items()
        if value not in (None, "") and key not in {"sample_setups"}
    )
    return {
        "title": label,
        "eyebrow": "How to read this research row",
        "summary": "This is an aggregate measurement used to judge selection quality. It does not define an entry price.",
        "steps": [
            "Read the side, horizon, and sample count before comparing the result.",
            "Use positive, sufficiently sampled rows to prioritize review—not to manufacture a trade.",
            "For execution, return to a concrete setup row and wait for its completed-bar trigger, stop, and target plan.",
        ],
        "evidence": [evidence or "No additional measurement fields are available."],
        "caution": _sample_caution(samples, research_only=True),
    }


def build_research_explanation(kind: str, row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a stable explanation contract for one GUI research row."""

    item: Mapping[str, Any] = row if isinstance(row, Mapping) else {}
    normalized = _text(kind).lower()
    if normalized.startswith("setup_") and normalized not in {
        "setup_scan_factor",
        "setup_tier_performance",
        "setup_catch_rate",
        "setup_human_pick",
    }:
        return _setup_explanation(normalized, item)
    if normalized.startswith("daytrade_"):
        return _daytrade_explanation(normalized, item)
    if normalized == "move_forensics":
        return _forensics_explanation(item)
    return _generic_research_explanation(normalized, item)


def build_plain_english_whats_working(
    *,
    current_rows: Sequence[Mapping[str, Any]] = (),
    short_term_rows: Sequence[Mapping[str, Any]] = (),
    recent_rows: Sequence[Mapping[str, Any]] = (),
    playbook_rows: Sequence[Mapping[str, Any]] = (),
    short_term_min_samples: int = 6,
) -> dict[str, Any]:
    """Summarize qualified measured leaders without filling empty slots."""

    bullets: list[str] = []
    ready = [row for row in current_rows if _text(row.get("tier")).upper() in {"S", "A"}]
    if ready:
        names = ", ".join(_text(row.get("symbol")).upper() for row in ready[:5] if row.get("symbol"))
        bullets.append(f"The current best-of-the-best list has {len(ready)} S/A setup(s): {names or 'see the table' }.")
    else:
        bullets.append("No current setup has cleared the S/A quality gate. An empty ready list is the honest result.")

    short_candidates = [
        row
        for row in short_term_rows
        if _int(row.get("samples_2d")) >= short_term_min_samples and (_float(row.get("avg_r_2d")) or 0.0) > 0
    ]
    if short_candidates:
        best = max(short_candidates, key=lambda row: _float(row.get("avg_r_2d")) or -1e9)
        bullets.append(
            f"For the first two sessions, {_text(best.get('side')).upper()} {_text(best.get('setup_family'))} "
            f"is strongest: {_signed(best.get('avg_r_2d'), 'R')} average, "
            f"{_percent(best.get('win_rate_2d'))} wins, n={_int(best.get('samples_2d'))}."
        )
    else:
        bullets.append("No short-term setup has both positive two-session R and the minimum sample floor yet.")

    swing_candidates = [
        row
        for row in recent_rows
        if _int(row.get("closed_setups")) >= 3 and (_float(row.get("avg_closed_r")) or 0.0) > 0
    ]
    if swing_candidates:
        best = max(swing_candidates, key=lambda row: _float(row.get("avg_closed_r")) or -1e9)
        bullets.append(
            f"Among recently closed swings, {_text(best.get('side')).upper()} {_text(best.get('setup_family'))} "
            f"leads at {_signed(best.get('avg_closed_r'), 'R')} average with "
            f"{_percent(best.get('target_hit_rate'))} target hits across {_int(best.get('closed_setups'))} closes."
        )
    else:
        bullets.append("No recent swing family has at least three closes and positive average R yet.")

    play_candidates = [
        row
        for row in playbook_rows
        if _int(row.get("closed_setups")) >= 5 and (_float(row.get("robust_closed_r")) or 0.0) > 0
    ]
    if play_candidates:
        best = max(play_candidates, key=lambda row: _float(row.get("robust_closed_r")) or -1e9)
        bullets.append(
            f"The best measured execution variant is {_text(best.get('side')).upper()} "
            f"{_text(best.get('setup_family'))}: stop at {_text(best.get('stop_reference_label'), 'its setup level')}, "
            f"then {_text(best.get('profit_take_summary'), 'follow the measured exit plan')} "
            f"({_signed(best.get('robust_closed_r'), 'R')} robust, n={_int(best.get('closed_setups'))})."
        )

    return {
        "headline": "What is working best, in plain English",
        "bullets": bullets,
        "caution": (
            "These are descriptive leaders from completed outcomes. They prioritize attention; they do not "
            "change scanner scores or turn an unqualified stock into a trade."
        ),
    }
