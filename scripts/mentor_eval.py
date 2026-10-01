"""Measure how well the Trade Mentor fetches the right data for plain questions.

    .venv\\Scripts\\python.exe scripts\\mentor_eval.py            (offline, the default)
    .venv\\Scripts\\python.exe scripts\\mentor_eval.py --live     (the real brain; the trader runs it)

The fixture ``tests/fixtures/mentor_eval_questions.json`` holds 60 questions written the way
the trader talks, each with ``expected_packs`` and ``must_mention``; ``simple: true`` marks the
ones a reply should answer in a few sentences.

``--offline`` runs only ``attach.plan_attachments`` (pure, no model, no store) and reports
attach recall per question: the share of expected packs the app attaches by itself.

``--live`` runs every question through the real brain on the app's endpoint (the mentor
tunnel, port 11436 by default, must be up and the GPU not handed to the night). It reports
the tool hit rate (every expected pack fetched, by the app or by the model), checklist
coverage on pre-trade questions (sections the model cited, and after the app's "Not
covered" appendix), must-mention rate, first-token p50/p95, and the style score (P14): reply
chars p50, the share of replies with headers, with a closing question or offer, and with
unasked context (regime/breadth/SPY pause on a question with no market or pre-trade cue),
and the style pass rate (no header, no offer, <= 600 chars on simple questions), for the
model's own words and after the app's guard.

``--rescore REPORT`` recomputes the style score of an earlier ``--live`` report (no model). It reads the live stores
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
#: P18: what a journal statement is expected to produce offline (a stored line, no pack).
JOURNAL = "journal_entry"


def load_fixture(path: Path | str = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now(fixture: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(str(fixture.get("now") or datetime.now(timezone.utc).isoformat()))


def offline_report(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Attach recall per question and overall (mean over questions)."""
    from mentor_app import journal_mode
    from mentor_app.attach import plan_attachments

    now, known = _now(fixture), dict(fixture.get("known_symbols") or {})
    # P16: the fixture's open book (a subset of the universe); a book question reads only these names.
    book = [str(sym).upper() for sym in fixture.get("book") or ()]
    rows = []
    for item in fixture.get("questions") or ():
        # P18: self talk is kept as a journal line and answered "Noted" with no packs; a question the app
        # took for self talk would never be answered, so it scores zero.
        planned = [request.name for request in plan_attachments(item["q"], known, now, book=book)]
        kind = journal_mode.classify(item["q"], planned=bool(planned))
        noted = kind.statement and not kind.asks
        if item.get("journal"):
            rows.append({"q": item["q"], "expected": [JOURNAL], "attached": [JOURNAL] if noted else [],
                         "missed": [] if noted else [JOURNAL], "recall": 1.0 if noted else 0.0, "journal": True})
            continue
        expected = list(item.get("expected_packs") or ())
        got = [] if noted else planned
        hit = [name for name in expected if name in got]
        rows.append({"q": item["q"], "expected": expected, "attached": got, "missed": [n for n in expected if n not in got],
                     "recall": len(hit) / len(expected) if expected else (0.0 if noted else 1.0)})
    recall = statistics.fmean(row["recall"] for row in rows) if rows else 0.0
    return {"mode": "offline", "questions": len(rows), "attach_recall": round(recall, 4), "rows": rows}


def style_summary(rows: list[Mapping[str, Any]], fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Adds ``style`` / ``style_after_app`` / ``style_pass`` to every answered row; returns the overall numbers."""
    from mentor_app import attach, style

    simple = {str(item["q"]) for item in fixture.get("questions") or () if item.get("simple")}
    known, now = dict(fixture.get("known_symbols") or {}), _now(fixture)
    book = [str(sym).upper() for sym in fixture.get("book") or ()]
    answered = [row for row in rows if "error" not in row]
    for row in answered:
        reply = str(row.get("reply") or "")
        row["style"] = style.measure(reply, row["q"])
        # P20: a summary ask is guarded as the app guards it (plain: no bullets, no bold).
        plain = attach.summary_turn(row["q"], attach.plan_attachments(row["q"], known, now, book=book))
        row["style_after_app"] = style.measure(style.guard(reply, plain=plain)[0], row["q"])
        row["simple"] = row["q"] in simple
        row["style_pass"] = style.passes(row["style"], simple=row["simple"])
        row["style_pass_after_app"] = style.passes(row["style_after_app"], simple=row["simple"])
    if not answered:
        return {}

    def share(rows_: list[Mapping[str, Any]], test: Any) -> float | None:
        return round(sum(1 for row in rows_ if test(row)) / len(rows_), 4) if rows_ else None

    plain = [row for row in answered if not row["style"]["market_cue"]]
    simple_rows = [row for row in answered if row["simple"]]
    return {
        "reply_chars_p50": _percentile([float(row["style"]["chars"]) for row in answered], 50),
        "headers_share": share(answered, lambda row: row["style"]["headers"] > 0),
        "offer_or_closing_share": share(answered, lambda row: row["style"]["offer_phrases"] > 0
                                        or row["style"]["closing_question"]),
        "unasked_context_share": share(plain, lambda row: row["style"]["context_lines_unasked"] > 0),
        "style_pass_rate": share(simple_rows, lambda row: row["style_pass"]),
        "style_pass_rate_after_app": share(simple_rows, lambda row: row["style_pass_after_app"]),
        "simple_questions": len(simple_rows),
    }


def rescore(report: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    """An earlier --live report's style score, recomputed (no model, nothing written)."""
    rows = [dict(row) for row in report.get("rows") or ()]
    return {"mode": "rescore", "style": style_summary(rows, fixture), "rows": rows}


def _style_line(summary: Mapping[str, Any]) -> str:
    return (f"style: pass {summary.get('style_pass_rate')} on {summary.get('simple_questions')} simple "
            f"({summary.get('style_pass_rate_after_app')} after app)  chars p50 {summary.get('reply_chars_p50')}  "
            f"headers {summary.get('headers_share')}  offers/closing {summary.get('offer_or_closing_share')}  "
            f"unasked context {summary.get('unasked_context_share')}")


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
        if item.get("journal"):
            continue  # P18: a journal statement gets "Noted" from the app, never a model turn
        chat = ChatModel()
        chat.add("user", item["q"])
        messages = chat.messages(context_text=context.as_text(), budget_tokens=settings.context_tokens())
        requests = attach.plan_attachments(item["q"], known, now, book=attach.book_symbols(context.rows))
        try:
            result = brain.run_turn(messages, model=model, endpoint=endpoint, keep_alive=settings.keep_alive(),
                                    num_ctx=settings.context_tokens(), tools=registry.tool_schemas(),
                                    native_tools=native, attachments=requests, question=item["q"],
                                    **attach.turn_shape(item["q"], requests))
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
        "style": style_summary(rows, fixture),
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
    mode.add_argument("--rescore", metavar="REPORT", help="the style score of an earlier --live JSON report")
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
        print(_style_line(report["style"]))
        print(f"report: {report['written']}")
        return 0
    if args.rescore:
        again = rescore(json.loads(Path(args.rescore).read_text(encoding="utf-8")), fixture)
        for row in again["rows"]:
            if "style" in row:
                mark = "ok  " if row["style_pass"] else "FAIL"
                print(f"{mark} {row['style']['chars']:>5}  {row['q']}")
        print(_style_line(again["style"]))
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
