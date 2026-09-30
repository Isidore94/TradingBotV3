"""Measure how well the Trade Mentor fetches the right data for plain questions.

    .venv\\Scripts\\python.exe scripts\\mentor_eval.py            (offline, the default)
    .venv\\Scripts\\python.exe scripts\\mentor_eval.py --live     (the real brain; the trader runs it)

The fixture ``tests/fixtures/mentor_eval_questions.json`` holds 40 questions written the way
the trader talks, each with ``expected_packs`` and ``must_mention``.

``--offline`` runs only ``attach.plan_attachments`` (pure, no model, no store) and reports
attach recall per question: the share of expected packs the app attaches by itself.

``--live`` runs every question through the real brain on the app's endpoint (the mentor
tunnel, port 11436 by default, must be up and the GPU not handed to the night). It reports
the tool hit rate (every expected pack fetched, by the app or by the model), checklist
coverage on pre-trade questions (sections the model cited, and after the app's "Not
covered" appendix), must-mention rate, and first-token p50/p95. It reads the live stores
read-only, writes nothing but its JSON report (``--out``, default the temp folder), and is
never run by pytest.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT_DIR / "tests" / "fixtures" / "mentor_eval_questions.json"
#: Pre-trade questions: those whose expected packs include the gate.
GATE = "gate_pack"


def load_fixture(path: Path | str = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now(fixture: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(str(fixture.get("now") or datetime.now(timezone.utc).isoformat()))


def offline_report(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Attach recall per question and overall (mean over questions)."""
    from mentor_app.attach import plan_attachments

    now, known = _now(fixture), dict(fixture.get("known_symbols") or {})
    rows = []
    for item in fixture.get("questions") or ():
        expected = list(item.get("expected_packs") or ())
        got = [request.name for request in plan_attachments(item["q"], known, now)]
        hit = [name for name in expected if name in got]
        rows.append({"q": item["q"], "expected": expected, "attached": got, "missed": [n for n in expected if n not in got],
                     "recall": len(hit) / len(expected) if expected else 1.0})
    recall = statistics.fmean(row["recall"] for row in rows) if rows else 0.0
    return {"mode": "offline", "questions": len(rows), "attach_recall": round(recall, 4), "rows": rows}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def live_report(fixture: Mapping[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Every question through the real brain; nothing written but the report."""
    from mentor_app import attach, brain, checklist, settings
    from mentor_app.chat_model import ChatModel
    from mentor_packs import context_pack, journal_pack, registry

    blocked = settings.gpu_block_reason()
    if blocked:
        raise SystemExit(f"Not now: {blocked}")
    endpoint = settings.endpoint()
    model = settings.mentor_model(lambda tag: brain.model_present(endpoint, tag, post=brain.default_post))
    native = brain.native_tools_for(brain.model_capabilities(endpoint, model, post=brain.default_post))
    context = context_pack.build()
    known = attach.known_symbols(context.rows, (), journal_pack.recent_symbols())
    for sym, side in (fixture.get("known_symbols") or {}).items():
        known.setdefault(str(sym).upper(), str(side or ""))
    now = datetime.now(timezone.utc)
    rows, firsts = [], []
    for item in fixture.get("questions") or ():
        chat = ChatModel()
        chat.add("user", item["q"])
        messages = chat.messages(context_text=context.as_text(), budget_tokens=settings.context_tokens())
        requests = attach.plan_attachments(item["q"], known, now)
        try:
            result = brain.run_turn(messages, model=model, endpoint=endpoint, keep_alive=settings.keep_alive(),
                                    num_ctx=settings.context_tokens(), tools=registry.tool_schemas(),
                                    native_tools=native, attachments=requests, question=item["q"])
        except Exception as exc:  # noqa: BLE001 - one failed question is reported, the run goes on
            rows.append({"q": item["q"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        fetched = {a["name"] for a in result["attached"] if not a.get("dropped")}
        fetched |= {str(call.get("name")) for call in result["tool_calls"]}
        expected = list(item.get("expected_packs") or ())
        text = str(result.get("text") or "")
        row: dict[str, Any] = {
            "q": item["q"], "expected": expected, "auto": sorted(a["name"] for a in result["attached"]),
            "model_calls": [call.get("name") for call in result["tool_calls"]],
            "hit": all(name in fetched for name in expected), "first_token_ms": result.get("first_token_ms"),
            "total_ms": result.get("total_ms"), "attach_ms": result.get("attach_ms"),
            "prompt_tokens": result.get("prompt_tokens"), "reply": text,
            "mentions": [w for w in item.get("must_mention") or () if w.lower() in text.lower()],
        }
        if GATE in expected:
            by_model = checklist.covered(text)
            after = by_model | set(checklist.missing(text) if result.get("appendix") else ())
            row["checklist_model"] = sorted(by_model)
            row["checklist_after_app"] = sorted(after)
        if result.get("first_token_ms") is not None:
            firsts.append(float(result["first_token_ms"]))
        rows.append(row)
    answered = [row for row in rows if "error" not in row]
    gates = [row for row in answered if "checklist_after_app" in row]
    sections = len(checklist.SECTIONS)
    report = {
        "mode": "live", "model": model, "native_tools": native, "endpoint": endpoint,
        "questions": len(rows), "errors": len(rows) - len(answered),
        "tool_hit_rate": round(sum(row["hit"] for row in answered) / len(answered), 4) if answered else None,
        "checklist_coverage_model": round(statistics.fmean(len(r["checklist_model"]) / sections for r in gates), 4)
        if gates else None,
        "checklist_coverage_after_app": round(statistics.fmean(len(r["checklist_after_app"]) / sections
                                                               for r in gates), 4) if gates else None,
        "first_token_p50_ms": _percentile(firsts, 50), "first_token_p95_ms": _percentile(firsts, 95),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"mentor_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written"] = str(target)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="attach recall only (default)")
    mode.add_argument("--live", action="store_true", help="the real brain on the mentor endpoint")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "mentor_eval"),
                        help="where --live writes its JSON report")
    args = parser.parse_args(argv)
    fixture = load_fixture(args.fixture)
    if args.live:
        report = live_report(fixture, out_dir=Path(args.out))
        print(f"model {report['model']} (tools: {'native' if report['native_tools'] else 'fallback'})")
        print(f"tool hit rate {report['tool_hit_rate']}  checklist {report['checklist_coverage_model']} (model) / "
              f"{report['checklist_coverage_after_app']} (after app)  first token p50 {report['first_token_p50_ms']} ms "
              f"p95 {report['first_token_p95_ms']} ms  errors {report['errors']}")
        print(f"report: {report['written']}")
        return 0
    report = offline_report(fixture)
    for row in report["rows"]:
        mark = "ok " if not row["missed"] else "MISS"
        print(f"{mark} {row['recall']:.2f}  {row['q']}" + (f"  (missed {', '.join(row['missed'])})" if row["missed"] else ""))
    print(f"attach recall {report['attach_recall']:.1%} over {report['questions']} questions")
    return 0


if __name__ == "__main__":
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    raise SystemExit(main())
