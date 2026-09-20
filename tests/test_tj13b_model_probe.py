"""TJ-13B item 1: the large local model is MEASURED before anything uses it.

`plan.md` §12.4 TJ-13 item 7, decision 0021 answers 19-20, decision 0018's
2026-09-19 amendment.

`ai_local_model_large` is configured on this desk
(`hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M`, verified 2026-09-19 in
`%LOCALAPPDATA%\\TradingBotV3\\local_settings.json`) and **nothing has ever run on
it**: in 476 live ledger rows the only models named are `gemma3:12b`,
`gemma3:12b-tbv3ctx` and `gemma3:12b-tbv3ctx-64k`. So the week story's
`reserve_minutes` cannot be written down by anybody - it has to be measured on
this box first, and the measurement has to be a durable record rather than a
number in a commit message.

**NO MODEL IS LOADED BY THESE TESTS.** Every endpoint here is a fake `post`
callable; a test that reached 127.0.0.1:11434 would load a second model beside
the nightly 12B run and is exactly what this file exists to make unnecessary.

Contract pinned by these tests, for the builder::

    ai_jobs.model_probe.PROBE_JOB == "model_probe"      # its own ledger row kind
    ai_jobs.model_probe.run_model_probe(
        *, tier="large", packs_root=None, now=None, force=False,
        post=<requests.post>, ledger_path=None) -> dict
            {"status": <ai_jobs.ledger status>, "reason": str,
             "measurement": dict | None}

    measurement keys: model, tier, packs, load_seconds, tokens_per_second,
                      peak_memory_mb, context_tokens_accepted

    run_ai_jobs.main(["--probe-model", "large"]) -> 0 measured, 1 refused/failed
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from zoneinfo import ZoneInfo  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")

#: The live window, read from the desk's own settings on 2026-09-19:
#: `ai_offhours_start` 01:00 ET, `ai_offhours_end` 09:00 ET = 22:00-06:00 PDT.
LIVE_START = "01:00"
LIVE_END = "09:00"

#: Saturday 2026-09-19 23:00 PDT is 02:00 ET on the 20th - inside the window,
#: on the night whose slate carries the week story. Its session date is Friday's.
SATURDAY_NIGHT = datetime(2026, 9, 19, 23, 0, tzinfo=PACIFIC)
#: The same Saturday at 14:00 PDT: 17:00 ET, outside the window, on the one
#: afternoon the trader is at the desk all day.
SATURDAY_AFTERNOON = datetime(2026, 9, 19, 14, 0, tzinfo=PACIFIC)
WEEKEND_SESSION = "2026-09-18"

ENDPOINT = "http://127.0.0.1:11434/v1"
LARGE_TAG = "hf.co/bartowski/google_gemma-3-27b-it-GGUF:Q3_K_M"
MEDIUM_TAG = "gemma3:12b-tbv3ctx-64k"

#: The week the probe must read: the five sessions 2026-09-14 … 2026-09-18
#: (`evidence_stats.WEEK_SESSIONS` is 5). The two packs a fortnight earlier are
#: the control - a probe that reads the whole store reads 7.
WEEK_PACKS = ("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18")
OLD_PACKS = ("2026-09-03", "2026-09-04")


def _settings(**values):
    """Patch BOTH settings seams: the window/store read `project_paths`, the
    model tags and endpoint are read through `ai_summary`'s own import."""
    import ai_summary
    import project_paths

    def _get(key, default=None):
        return values.get(key, default)

    return _MultiPatch(
        mock.patch.object(ai_summary, "get_local_setting", _get),
        mock.patch.object(project_paths, "get_local_setting", _get),
    )


class _MultiPatch:
    def __init__(self, *patches):
        self._patches = patches

    def __enter__(self):
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        return False


def _live_settings(**extra):
    values = {
        "ai_offhours_start": LIVE_START,
        "ai_offhours_end": LIVE_END,
        "ai_local_endpoint_url": ENDPOINT,
        "ai_local_model_large": LARGE_TAG,
        "ai_local_model_medium": MEDIUM_TAG,
        "ai_local_context_tokens": 65536,
    }
    values.update(extra)
    return _settings(**values)


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _valid_reply_text() -> str:
    """A model answer that satisfies the closed contract without citing
    anything the probe's own packs do not contain."""
    import ai_summary

    sections = {name: [] for name in ai_summary.MODEL_SUMMARY_SECTIONS}
    return json.dumps(
        {
            "executive_summary": "One measured week, narrated from the packs.",
            **sections,
        }
    )


#: What the fake server reports it actually ingested. The probe must record THIS
#: as the context it accepted, not the number the desk configured.
SERVER_PROMPT_TOKENS = 41_234
SERVER_COMPLETION_TOKENS = 812


def _fake_post(calls, *, text=None, usage=None):
    def _post(url, **kwargs):
        calls.append((url, kwargs))
        payload = {
            "id": "chatcmpl-probe",
            "choices": [
                {
                    "message": {"role": "assistant", "content": text or _valid_reply_text()},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage
            if usage is not None
            else {
                "prompt_tokens": SERVER_PROMPT_TOKENS,
                "completion_tokens": SERVER_COMPLETION_TOKENS,
                "total_tokens": SERVER_PROMPT_TOKENS + SERVER_COMPLETION_TOKENS,
            },
        }
        return _Response(payload)

    return _post


def _never_post(*args, **kwargs):
    raise AssertionError(
        "the probe started a model request when it must not have: a model load "
        "is exactly what the night-only rule is about"
    )


def _write_packs(root: Path) -> Path:
    """One week of daily fact packs, plus two from a fortnight earlier.

    The layout is the store's own (`<digests>/facts/<session>.json`), so the
    probe can read it with `ai_jobs.digest.read_fact_pack_files`.
    """
    facts = root / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    for day in (*WEEK_PACKS, *OLD_PACKS):
        (facts / f"{day}.json").write_text(
            json.dumps(
                {
                    "schema": "ai_daily_fact_pack_v1",
                    "session_date": day,
                    "generated_at": f"{day}T23:10:00-04:00",
                    "headline": f"probe marker {day}",
                    "sections": {"decisions": {"n": 4}},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return root


def _snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _probe_rows(path: Path) -> list[dict]:
    from ai_jobs import model_probe

    return [row for row in _rows(path) if row.get("job") == model_probe.PROBE_JOB]


# ---------------------------------------------------------------------------
# the command exists, and it runs only the probe
# ---------------------------------------------------------------------------


def test_probe_model_large_is_a_command_that_runs_no_slate(monkeypatch):
    """`--probe-model large` is a measurement, not a night.

    It must not build a slate on its way: an unpinned entry point on a Saturday
    evening would otherwise assemble the real Saturday slate and start
    `ai_summary` beside the probe.
    """
    import run_ai_jobs
    from ai_jobs import model_probe, runner

    def _no_slate(*args, **kwargs):
        raise AssertionError("--probe-model must not run the night's slate")

    monkeypatch.setattr(runner, "run_slots", _no_slate)
    seen: list[dict] = []

    def _fake_probe(**kwargs):
        seen.append(dict(kwargs))
        from ai_jobs import ledger

        return {"status": ledger.STATUS_MANUAL, "reason": "measured", "measurement": {}}

    monkeypatch.setattr(model_probe, "run_model_probe", _fake_probe)

    assert run_ai_jobs.main(["--probe-model", "large"]) == 0
    assert len(seen) == 1
    assert seen[0].get("tier") == "large"


def test_a_refused_probe_exits_nonzero_so_the_operator_is_told(monkeypatch):
    """"It printed something and exited 0" and "it measured the model" must not
    look alike to whoever typed the command."""
    import run_ai_jobs
    from ai_jobs import ledger, model_probe

    monkeypatch.setattr(
        model_probe,
        "run_model_probe",
        lambda **kwargs: {
            "status": ledger.STATUS_SKIPPED,
            "reason": "outside the off-hours window (01:00-09:00 ET)",
            "measurement": None,
        },
    )

    assert run_ai_jobs.main(["--probe-model", "large"]) == 1


# ---------------------------------------------------------------------------
# it is a model load, so it is night-only - and --force does not buy the clock
# ---------------------------------------------------------------------------


def test_a_daytime_probe_is_refused_and_force_does_not_buy_the_clock(tmp_path):
    """TJ-13A item 1's rule, applied to the heaviest load on the box.

    14:00 on a Saturday is precisely the moment the trader is using the desk
    (decision 0021 answer 19), and a 27B load is the thing the rule is about.
    """
    from ai_jobs import ledger, model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_AFTERNOON,
            force=True,
            post=_never_post,
            ledger_path=led,
        )

    assert result["status"] == ledger.STATUS_SKIPPED
    assert result["measurement"] is None
    assert "window" in result["reason"].lower()
    # Nothing it wrote may read as a measurement or as coverage.
    assert all(not row.get("model_probe") for row in _probe_rows(led))
    assert model_probe.PROBE_JOB not in ledger.completed_jobs(WEEKEND_SESSION, path=led)


# ---------------------------------------------------------------------------
# what a night probe records
# ---------------------------------------------------------------------------


def test_a_night_probe_records_load_time_throughput_memory_and_the_accepted_context(tmp_path):
    """The four numbers the week slot's reserve has to come from.

    `context_tokens_accepted` is what the SERVER reported it ingested, never the
    `ai_local_context_tokens` setting: a local server silently truncates and
    answers from what survived, and the whole point of measuring the 27B is to
    learn what it really takes.
    """
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    calls: list[tuple[str, dict]] = []

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_fake_post(calls),
            ledger_path=led,
        )

    assert calls, "the probe never asked the endpoint anything"
    measurement = result["measurement"]
    assert measurement is not None, result["reason"]

    assert measurement["model"] == LARGE_TAG
    assert measurement["tier"] == "large"
    assert measurement["context_tokens_accepted"] == SERVER_PROMPT_TOKENS
    assert float(measurement["load_seconds"]) > 0
    assert float(measurement["tokens_per_second"]) > 0
    assert float(measurement["peak_memory_mb"]) > 0

    rows = _probe_rows(led)
    assert len(rows) == 1, "one probe, one row"
    row = rows[0]
    assert row["model"] == LARGE_TAG
    assert row["session_date"] == WEEKEND_SESSION
    assert row["model_probe"]["context_tokens_accepted"] == SERVER_PROMPT_TOKENS
    assert float(row["model_probe"]["tokens_per_second"]) > 0


def test_the_probe_row_never_counts_as_session_coverage(tmp_path):
    """Its own row kind, and not one that says the session was covered.

    A probe is a measurement the operator asked for. If it counted, a Saturday
    probe would tell the next firing that Friday's session had been served.
    """
    from ai_jobs import ledger, model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_fake_post([]),
            ledger_path=led,
        )

    rows = _probe_rows(led)
    assert rows, "the probe wrote no row at all"
    assert rows[0]["job"] == model_probe.PROBE_JOB
    assert rows[0]["status"] not in ledger.CANONICAL_COMPLETION_STATUSES
    assert model_probe.PROBE_JOB not in ledger.completed_jobs(WEEKEND_SESSION, path=led)
    # ...and it never claims to be one of the night's slots either.
    assert "ai_summary" not in {row["job"] for row in _rows(led)}


def test_the_probe_reads_one_week_of_packs_and_writes_nothing_into_the_store(tmp_path):
    """Five session packs, not the whole store - and the source is untouched.

    Seven packs sit in the fixture; the two from a fortnight earlier are the
    control. `evidence_stats.WEEK_SESSIONS` is 5, which is the week Weekend Prep
    means everywhere else.
    """
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    before = _snapshot(packs)
    calls: list[tuple[str, dict]] = []

    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_fake_post(calls),
            ledger_path=led,
        )

    assert result["measurement"]["packs"] == 5
    prompt = " ".join(
        str(message.get("content") or "")
        for _url, kwargs in calls
        for message in kwargs["json"]["messages"]
    )
    assert "2026-09-18" in prompt, "it did not send the week it claims to have read"
    assert "2026-09-03" not in prompt, "it read outside the week"
    # The packs are read; nothing is written back over them, and no file the
    # probe invented appears beside them.
    assert _snapshot(packs) == before


def test_the_probe_asks_the_configured_large_model_never_the_medium_one(tmp_path):
    """The tag comes from `ai_local_model_large`, through settings, never code."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    calls: list[tuple[str, dict]] = []

    with _live_settings(ai_local_model_large="gemma3:27b-tbv3-probe"):
        model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_fake_post(calls),
            ledger_path=led,
        )

    models = [kwargs["json"]["model"] for _url, kwargs in calls]
    assert models == ["gemma3:27b-tbv3-probe"]
    assert MEDIUM_TAG not in models


# ---------------------------------------------------------------------------
# a dead endpoint, and the artifacts that were already there
# ---------------------------------------------------------------------------


def test_an_unreachable_endpoint_degrades_in_seconds_and_keeps_every_prior_row(tmp_path):
    """TJ-13A item 3's rule for the heaviest call on the box.

    A refused endpoint is a different fact from a bad answer, and it costs ONE
    call - measured 2.06 s on 2026-09-19. A probe that retried into the night
    would be the 53-slice defect again, with a 27B in front of it.
    """
    import ai_summary
    from ai_jobs import ledger, model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    # Two rows written before the probe: they are the "prior artifacts".
    ledger.record(
        job="daily_digest",
        status=ledger.STATUS_OK,
        session_date=WEEKEND_SESSION,
        reason="fact pack for 2026-09-18",
        path=led,
    )
    ledger.record(
        job="market_story_narration",
        status=ledger.STATUS_OK,
        session_date=WEEKEND_SESSION,
        reason="weekly narration published",
        path=led,
    )
    before = led.read_text(encoding="utf-8")

    attempts: list[str] = []

    def _refused(url, **kwargs):
        attempts.append(url)
        raise ConnectionError("[WinError 10061] No connection could be made")

    started = time.perf_counter()
    with _live_settings():
        result = model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_refused,
            ledger_path=led,
        )
    elapsed = time.perf_counter() - started

    assert result["status"] == ledger.STATUS_FAILED
    assert result["measurement"] is None
    assert len(attempts) == 1, "a dead endpoint costs one call, not a retry ladder"
    assert elapsed < 5.0, f"the probe took {elapsed:.1f}s to discover a refused endpoint"
    # Named by the class the repair introduced, not by a generic message.
    assert ai_summary.is_endpoint_unreachable(
        ai_summary.LocalEndpointUnreachable(result["reason"])
    )
    assert ai_summary.LOCAL_UNREACHABLE_MARKER in result["reason"]

    # Every prior row is exactly as it was, and the failure is one appended row.
    assert led.read_text(encoding="utf-8").startswith(before)
    assert len(_probe_rows(led)) == 1
    assert _probe_rows(led)[0]["status"] == ledger.STATUS_FAILED


# ---------------------------------------------------------------------------
# the contract the answer is validated against (plan.md §12.3)
# ---------------------------------------------------------------------------


def _schema_nodes(node, path="$"):
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from _schema_nodes(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _schema_nodes(value, f"{path}[{index}]")


def test_the_probe_sends_a_closed_schema_that_asks_for_citations(tmp_path):
    """Ground rule: every model output is validated against a CLOSED JSON schema
    whose text fields cite allowed `source_id`s - and no field carries a
    `maxLength` of exactly 2,000, which is the grammar-compile defect behind
    gate #144."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")
    calls: list[tuple[str, dict]] = []

    with _live_settings():
        model_probe.run_model_probe(
            tier="large",
            packs_root=packs,
            now=SATURDAY_NIGHT,
            post=_fake_post(calls),
            ledger_path=led,
        )

    payload = calls[0][1]["json"]
    schema = payload["response_format"]["json_schema"]["schema"]

    property_names: set[str] = set()
    for path, node in _schema_nodes(schema):
        if "properties" in node:
            assert node.get("additionalProperties") is False, f"{path} is an open object"
            property_names.update(node["properties"])
        assert node.get("maxLength") != 2000, f"{path} carries the maxLength that will not compile"

    assert property_names & {"evidence_refs", "source_ids", "sources"}, (
        "the contract asks for no citations, so nothing the model writes can be "
        "traced to a source_id"
    )


def test_an_unknown_tier_is_refused_rather_than_silently_measured(tmp_path):
    """"large" and "medium" are the tiers that exist. A typo must not quietly
    measure the 12B and file the number under the 27B."""
    from ai_jobs import model_probe

    led = tmp_path / "ledger.jsonl"
    packs = _write_packs(tmp_path / "store" / "digests")

    with _live_settings():
        with pytest.raises(ValueError):
            model_probe.run_model_probe(
                tier="enormous",
                packs_root=packs,
                now=SATURDAY_NIGHT,
                post=_never_post,
                ledger_path=led,
            )
