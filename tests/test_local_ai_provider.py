"""Local inference endpoint plumbing (docs/LOCAL_AI_AUTOMATION_PLAN.md Phase 0).

Phase 0 adds an OpenAI-compatible ``local`` provider to the two existing AI
call sites and changes nothing else. The load-bearing assertions here are the
negative ones: with the new settings unset, the request each cloud provider
receives is byte-identical to what it received before this provider existed,
and no code path can reach a local endpoint.

Everything is offline - the fake transport is the only endpoint these tests
know about.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ENDPOINT = "http://127.0.0.1:11434/v1"


def _valid_summary(ref: str) -> dict:
    import ai_summary

    # Model sections only: data_quality is machine-owned and a model that
    # returns it is rejected (Sol 5.6 verification review, item 4).
    sections = {name: [] for name in ai_summary.MODEL_SUMMARY_SECTIONS}
    sections["what_is_working"] = [
        {"statement": "Swing rows are shown first.", "evidence_refs": [ref], "confidence": "high"}
    ]
    return {"executive_summary": "The selected evidence supports one measured finding.", **sections}


def _daily_overrides(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for source_id in ("daily.auto_report", "daily.market_prep", "daily.master_events"):
        path = tmp_path / (source_id.replace(".", "_") + ".txt")
        path.write_text(f"Evidence from {source_id}\n", encoding="utf-8")
        paths[source_id] = path
    return paths


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def _chat_response(text: str, *, status_code: int = 200, usage: dict | None = None):
    """A chat-completions body. ``usage`` omitted entirely unless asked for:
    some llama.cpp builds do not report it, and the default here must stay that
    shape so the truncation tripwire is exercised against its silent case."""
    payload = {
        "id": "chatcmpl-local",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    if usage is not None:
        payload["usage"] = usage
    return _Response(payload, status_code=status_code)


def _settings(**values):
    """Patch the local-settings reader for scripts/ai_summary.py."""
    import ai_summary

    return mock.patch.object(
        ai_summary,
        "get_local_setting",
        lambda key, default=None: values.get(key, default),
    )


class LocalProviderSelectionTests(unittest.TestCase):
    def test_local_provider_is_off_until_an_endpoint_is_configured(self):
        import ai_summary

        with _settings():
            self.assertFalse(ai_summary.local_provider_enabled())
            self.assertEqual(ai_summary.local_endpoint_url(), "")
        with _settings(ai_local_endpoint_url=ENDPOINT):
            self.assertTrue(ai_summary.local_provider_enabled())

    def test_normalize_provider_accepts_local_and_still_rejects_junk(self):
        import ai_summary

        self.assertEqual(ai_summary.normalize_provider("Local"), "local")
        self.assertEqual(ai_summary.normalize_provider("openai"), "openai")
        with self.assertRaises(ValueError):
            ai_summary.normalize_provider("ollama")

    def test_default_local_model_comes_from_settings_not_code(self):
        import ai_summary

        with _settings():
            self.assertEqual(ai_summary.default_model_for("local"), "gemma3:12b")
        with _settings(ai_local_model_medium="qwen3:14b"):
            self.assertEqual(ai_summary.default_model_for("local"), "qwen3:14b")
        # Cloud defaults are untouched by any local setting.
        with _settings(ai_local_model_medium="qwen3:14b"):
            self.assertEqual(ai_summary.default_model_for("openai"), ai_summary.DEFAULT_MODELS["openai"])

    def test_trailing_slash_in_the_endpoint_does_not_double_up(self):
        import ai_summary

        with _settings(ai_local_endpoint_url=ENDPOINT + "/"):
            self.assertEqual(ai_summary.local_endpoint_url(), ENDPOINT)


class CloudPathIsUnchangedTests(unittest.TestCase):
    """With the settings unset, cloud requests must be byte-identical."""

    def _capture(self, provider, evidence):
        import ai_summary

        summary_text = json.dumps(_valid_summary("daily.auto_report"))
        calls = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            if provider == "openai":
                return _Response(
                    {
                        "id": "resp-openai",
                        "output": [
                            {"type": "message", "content": [{"type": "output_text", "text": summary_text}]}
                        ],
                    }
                )
            return _Response({"id": "msg-anthropic", "content": [{"type": "text", "text": summary_text}]})

        result = ai_summary.request_ai_summary(
            provider=provider,
            model="test-model",
            api_key="super-secret",
            evidence=evidence,
            post=fake_post,
        )
        return result, calls

    def test_cloud_requests_are_identical_with_and_without_local_settings(self):
        import tempfile

        import ai_summary

        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                with tempfile.TemporaryDirectory() as raw:
                    # One package for both runs: its id and hash are stamped at
                    # build time, so rebuilding it would compare two different
                    # payloads and prove nothing about the provider plumbing.
                    evidence = ai_summary.build_evidence_package(
                        ["daily_report"], source_overrides=_daily_overrides(Path(raw))
                    )
                    with _settings():
                        _result, unset_calls = self._capture(provider, evidence)
                    with _settings(
                        ai_local_endpoint_url=ENDPOINT,
                        ai_local_model_medium="gemma3:12b",
                        ai_local_model_small="gemma3:4b",
                        ai_local_model_large="gemma3:27b",
                        # A deliberately tiny local budget: if it ever leaked
                        # into the cloud path it would starve the package and
                        # this byte-comparison would fail loudly.
                        ai_local_evidence_budget_chars=500,
                    ):
                        _result, set_calls = self._capture(provider, evidence)

                url_unset, kwargs_unset = unset_calls[0]
                url_set, kwargs_set = set_calls[0]
                self.assertEqual(url_unset, url_set)
                self.assertEqual(
                    json.dumps(kwargs_unset["json"], sort_keys=True),
                    json.dumps(kwargs_set["json"], sort_keys=True),
                )
                self.assertEqual(kwargs_unset["headers"], kwargs_set["headers"])
                # And the endpoint is still the real provider, not localhost.
                self.assertTrue(url_set.startswith("https://"))

    def test_cloud_providers_still_demand_a_key(self):
        import ai_summary

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(ValueError):
                ai_summary.request_ai_summary(
                    provider="openai", model="", api_key="", evidence={}, post=lambda *a, **k: None
                )


class LocalRequestTests(unittest.TestCase):
    def setUp(self):
        import ai_summary
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.evidence = ai_summary.build_evidence_package(
            ["daily_report"], source_overrides=_daily_overrides(Path(self._tmp.name))
        )
        self.summary_text = json.dumps(_valid_summary("daily.auto_report"))

    def test_local_request_uses_the_chat_completions_shape(self):
        import ai_summary

        calls = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            return _chat_response(self.summary_text)

        with _settings(ai_local_endpoint_url=ENDPOINT, ai_local_model_medium="gemma3:12b"):
            result = ai_summary.request_ai_summary(
                provider="local",
                model="",
                api_key="",
                evidence=self.evidence,
                post=fake_post,
            )

        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["provider"], "local")
        self.assertEqual(result["model"], "gemma3:12b")
        url, kwargs = calls[0]
        self.assertEqual(url, f"{ENDPOINT}/chat/completions")
        self.assertEqual([m["role"] for m in kwargs["json"]["messages"]], ["system", "user"])
        # No credential exists for a localhost server; the placeholder is not a secret.
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer local")

    def test_local_request_states_the_required_shape_in_the_prompt(self):
        """Regression: gemma3:12b answered with a bare {"summary": ...} object
        until the schema was spelled out. The cloud providers learn the shape
        from their structured-output contracts; a local server that ignores
        response_format has nothing else to go on."""
        import ai_summary

        calls = []

        with _settings(ai_local_endpoint_url=ENDPOINT):
            ai_summary.request_ai_summary(
                provider="local",
                model="gemma3:12b",
                api_key="",
                evidence=self.evidence,
                post=lambda url, **kwargs: (
                    calls.append(kwargs) or _chat_response(self.summary_text)
                ),
            )

        user_message = calls[0]["json"]["messages"][1]["content"]
        for key in ("executive_summary", *ai_summary.MODEL_SUMMARY_SECTIONS):
            self.assertIn(key, user_message)
        # ...and the machine-owned section is explicitly forbidden.
        self.assertIn("Do NOT return a data_quality section", user_message)
        self.assertIn("statement", user_message)
        self.assertIn("evidence_refs", user_message)
        # Best-effort structured output on top of the prompt, never relied on.
        response_format = calls[0]["json"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["schema"], ai_summary.AI_SUMMARY_JSON_SCHEMA
        )

    def test_the_shape_hint_is_local_only(self):
        """The shared prompt the cloud providers get must not have grown one."""
        import ai_summary

        shared = ai_summary._user_prompt(self.evidence)
        local = ai_summary._local_user_prompt(self.evidence)
        self.assertTrue(local.startswith(shared))
        self.assertNotIn("REQUIRED OUTPUT SHAPE", shared)
        self.assertIn("REQUIRED OUTPUT SHAPE", local)

    def test_local_output_is_validated_against_the_same_schema(self):
        import ai_summary

        bogus = json.dumps({"executive_summary": "hi"})  # missing every section

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(RuntimeError):
                ai_summary.request_ai_summary(
                    provider="local",
                    model="gemma3:12b",
                    api_key="",
                    evidence=self.evidence,
                    post=lambda url, **kwargs: _chat_response(bogus),
                )

    def test_local_output_citing_unknown_evidence_is_rejected(self):
        import ai_summary

        hallucinated = json.dumps(_valid_summary("daily.a_source_that_does_not_exist"))

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(RuntimeError):
                ai_summary.request_ai_summary(
                    provider="local",
                    model="gemma3:12b",
                    api_key="",
                    evidence=self.evidence,
                    post=lambda url, **kwargs: _chat_response(hallucinated),
                )

    def test_invalid_json_is_retried_exactly_once(self):
        import ai_summary

        responses = [_chat_response("not json at all"), _chat_response(self.summary_text)]
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return responses[len(calls) - 1]

        with _settings(ai_local_endpoint_url=ENDPOINT):
            result = ai_summary.request_ai_summary(
                provider="local",
                model="gemma3:12b",
                api_key="",
                evidence=self.evidence,
                post=fake_post,
            )

        self.assertEqual(result["status"], "validated")
        self.assertEqual(len(calls), 2)

    def test_persistently_invalid_json_fails_rather_than_looping(self):
        import ai_summary

        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return _chat_response("still not json")

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(RuntimeError):
                ai_summary.request_ai_summary(
                    provider="local",
                    model="gemma3:12b",
                    api_key="",
                    evidence=self.evidence,
                    post=fake_post,
                )

        self.assertEqual(len(calls), 2, "one retry, not an unbounded loop")

    def test_local_provider_without_an_endpoint_is_a_clean_error(self):
        import ai_summary

        with _settings():
            with self.assertRaises(RuntimeError) as ctx:
                ai_summary.request_ai_summary(
                    provider="local",
                    model="gemma3:12b",
                    api_key="",
                    evidence=self.evidence,
                    post=lambda *a, **k: _chat_response(self.summary_text),
                )
        self.assertIn("ai_local_endpoint_url", str(ctx.exception))

    def test_endpoint_down_is_a_clean_error_not_a_crash(self):
        import ai_summary

        def refuse(url, **kwargs):
            raise ConnectionError("connection refused")

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(RuntimeError) as ctx:
                ai_summary.request_ai_summary(
                    provider="local",
                    model="gemma3:12b",
                    api_key="",
                    evidence=self.evidence,
                    post=refuse,
                )
        self.assertIn("unreachable", str(ctx.exception))

    def test_http_error_from_the_local_server_is_surfaced(self):
        import ai_summary

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(RuntimeError) as ctx:
                ai_summary.request_ai_summary(
                    provider="local",
                    model="no-such-model",
                    api_key="",
                    evidence=self.evidence,
                    post=lambda url, **kwargs: _chat_response("", status_code=404),
                )
        self.assertIn("404", str(ctx.exception))


class MarketPrepBaseUrlTests(unittest.TestCase):
    """market_prep/services/ai_service.py: same switch, same default-off rule."""

    def _patch_settings(self, **values):
        from market_prep.services import ai_service

        return mock.patch.object(
            ai_service,
            "_read_local_setting",
            lambda key, default="": values.get(key, default),
        )

    def test_settings_unset_leaves_the_cloud_configuration_alone(self):
        from market_prep.models import MarketPrepConfig
        from market_prep.services.ai_service import get_market_prep_ai_settings

        with self._patch_settings():
            settings = get_market_prep_ai_settings(MarketPrepConfig())

        self.assertEqual(settings["base_url"], "")
        self.assertEqual(settings["model"], "gpt-5.2")

    def test_endpoint_setting_switches_the_call_site_to_local(self):
        from market_prep.models import MarketPrepConfig
        from market_prep.services.ai_service import get_market_prep_ai_settings

        with self._patch_settings(
            ai_local_endpoint_url=ENDPOINT + "/",
            ai_local_model_medium="gemma3:12b",
        ):
            settings = get_market_prep_ai_settings(MarketPrepConfig())

        self.assertEqual(settings["base_url"], ENDPOINT)
        self.assertEqual(settings["model"], "gemma3:12b")

    def test_an_explicit_config_model_still_wins(self):
        from market_prep.models import MarketPrepConfig
        from market_prep.services.ai_service import get_market_prep_ai_settings

        config = MarketPrepConfig(market_prep_ai={"model": "my-tuned-model"})
        with self._patch_settings(ai_local_endpoint_url=ENDPOINT):
            settings = get_market_prep_ai_settings(config)

        self.assertEqual(settings["model"], "my-tuned-model")

    def test_a_per_call_site_base_url_overrides_the_shared_setting(self):
        from market_prep.models import MarketPrepConfig
        from market_prep.services.ai_service import get_market_prep_ai_settings

        config = MarketPrepConfig(market_prep_ai={"base_url": "https://api.openai.com/v1"})
        with self._patch_settings(ai_local_endpoint_url=ENDPOINT):
            settings = get_market_prep_ai_settings(config)

        self.assertEqual(settings["base_url"], "https://api.openai.com/v1")

    def test_client_gets_no_base_url_when_unset(self):
        from market_prep.services import ai_service

        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"openai": mock.Mock(OpenAI=_FakeOpenAI)}):
            ai_service._build_openai_client({"timeout_seconds": 30, "base_url": ""}, "sk-cloud")

        self.assertNotIn("base_url", captured)
        self.assertEqual(captured["api_key"], "sk-cloud")

    def test_client_gets_the_base_url_when_set(self):
        from market_prep.services import ai_service

        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with mock.patch.dict(sys.modules, {"openai": mock.Mock(OpenAI=_FakeOpenAI)}):
            ai_service._build_openai_client({"timeout_seconds": 30, "base_url": ENDPOINT}, "local")

        self.assertEqual(captured["base_url"], ENDPOINT)

    def test_missing_key_falls_back_to_the_placeholder_only_for_local(self):
        from market_prep.services import ai_service

        with mock.patch.object(ai_service, "resolve_openai_api_key", lambda config: ""):
            self.assertEqual(ai_service._resolve_brief_api_key(None, {"base_url": ENDPOINT}), "local")
            self.assertEqual(ai_service._resolve_brief_api_key(None, {"base_url": ""}), "")

    def test_local_deployments_use_chat_completions_not_the_responses_api(self):
        from market_prep.services import ai_service

        client = mock.Mock()
        client.chat.completions.create.return_value = mock.Mock(
            choices=[mock.Mock(message=mock.Mock(content="  local brief  "))]
        )

        text = ai_service._generate_brief_text(
            client,
            {"base_url": ENDPOINT, "model": "gemma3:12b"},
            instructions="be concise",
            prompt="context",
        )

        self.assertEqual(text, "local brief")
        client.responses.create.assert_not_called()
        _args, kwargs = client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gemma3:12b")

    def test_cloud_deployments_still_use_the_responses_api(self):
        from market_prep.services import ai_service

        client = mock.Mock()
        client.responses.create.return_value = mock.Mock(output_text=" cloud brief ")

        text = ai_service._generate_brief_text(
            client,
            {"base_url": "", "model": "gpt-5.2"},
            instructions="be concise",
            prompt="context",
        )

        self.assertEqual(text, "cloud brief")
        client.chat.completions.create.assert_not_called()
        _args, kwargs = client.responses.create.call_args
        self.assertEqual(kwargs["model"], "gpt-5.2")
        self.assertEqual(kwargs["input"], "context")

    def test_brief_records_where_the_text_actually_came_from(self):
        from market_prep.services import ai_service

        self.assertEqual(ai_service._brief_source({"base_url": ENDPOINT}), "local")
        self.assertEqual(ai_service._brief_source({"base_url": ""}), "openai")


class LocalEvidenceBudgetTests(unittest.TestCase):
    """The evidence cap is per-call-site and never lowers the cloud ceiling.

    A local server truncates an over-long prompt in silence, which defeats the
    packager's own honest degradation. Capping the evidence is the fix; doing
    it globally would starve metered models that have room for far more.
    """

    def test_local_budget_defaults_and_can_be_configured(self):
        import ai_summary

        with _settings():
            self.assertEqual(
                ai_summary.evidence_budget_for("local"),
                ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS,
            )
        with _settings(ai_local_evidence_budget_chars=9000):
            self.assertEqual(ai_summary.evidence_budget_for("local"), 9000)

    def test_a_broken_budget_value_falls_back_instead_of_funding_nothing(self):
        import ai_summary

        # 0 would fund no sources at all, which looks exactly like a day with
        # no evidence -- the failure the budget exists to make visible.
        for broken in (0, -1, "", "lots", None, True):
            with self.subTest(value=broken):
                with _settings(ai_local_evidence_budget_chars=broken):
                    self.assertEqual(
                        ai_summary.evidence_budget_for("local"),
                        ai_summary.DEFAULT_LOCAL_EVIDENCE_BUDGET_CHARS,
                    )

    def test_the_cloud_ceiling_is_untouched_by_the_local_budget(self):
        import ai_summary

        with _settings(ai_local_evidence_budget_chars=500):
            for provider in ("openai", "anthropic"):
                with self.subTest(provider=provider):
                    self.assertEqual(
                        ai_summary.evidence_budget_for(provider),
                        ai_summary.MAX_TOTAL_EVIDENCE_CHARS,
                    )

    def test_the_default_budget_leaves_room_for_generation_and_the_retry(self):
        """The derivation, asserted rather than trusted.

        The retry re-sends the full evidence PLUS the validator's rejection, so
        a budget that only fits the first attempt turns every retry into the
        truncation it was meant to prevent. Estimated at the pessimistic
        3.0 chars/token, the worst-case prompt must still fit the context left
        after generation.
        """
        import tempfile

        import ai_summary

        # Read from the module, never re-typed here. The 2026-08-27 shear
        # happened because this arithmetic lived in a comment beside a
        # hand-carried number, and both its inputs were wrong by the same sign.
        context_window = ai_summary.DEFAULT_LOCAL_CONTEXT_TOKENS
        generation = ai_summary.LOCAL_GENERATION_TOKENS
        with tempfile.TemporaryDirectory() as raw:
            with _settings():
                evidence = ai_summary.build_evidence_package(
                    ["daily_report"],
                    source_overrides=_daily_overrides(Path(raw)),
                    budget_chars=ai_summary.evidence_budget_for("local"),
                )
            # A long rejection is the worst realistic retry.
            prompt = ai_summary._local_user_prompt(evidence, "rejected: " + ("x" * 900))
            scaffold = ai_summary._system_instruction()

        # At the WORST measured tokenization rate, not a flattering one: 2.06
        # chars/token was measured against the desk's own model, so dividing by
        # 3.0 here understated every real prompt by ~45%.
        estimated_tokens = (len(prompt) + len(scaffold)) / ai_summary._BUDGET_CHARS_PER_TOKEN
        self.assertLess(
            estimated_tokens,
            context_window - generation,
            f"worst-case retry prompt estimates at {int(estimated_tokens)} tokens, which does "
            f"not fit {context_window - generation} tokens of context left after generation",
        )

    def test_the_local_path_waits_longer_than_a_cloud_call_would(self):
        """A hosted API that has not answered in five minutes has failed. A
        local 12B has not failed, it is still working: measured 2026-08-28 at
        ~118 tok/s evaluating the prompt, so the nightly summary's own 45,302
        token package needs about six minutes before the first output token
        exists. The old 300s clamp turned that into a timeout."""
        import ai_summary

        seen = {}

        def fake_post(url, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            raise RuntimeError("stop here; the timeout is what is under test")

        with _settings(ai_local_endpoint_url="http://127.0.0.1:11434/v1"):
            with self.assertRaises(RuntimeError):
                ai_summary._request_local_summary(
                    model="m", api_key="k", evidence={"sources": []},
                    timeout_seconds=900, post=fake_post,
                )
        self.assertEqual(seen["timeout"], 900, "the caller's timeout must survive")

        with _settings(ai_local_endpoint_url="http://127.0.0.1:11434/v1"):
            with self.assertRaises(RuntimeError):
                ai_summary._request_local_summary(
                    model="m", api_key="k", evidence={"sources": []},
                    timeout_seconds=99_999, post=fake_post,
                )
        self.assertEqual(
            seen["timeout"],
            ai_summary.LOCAL_REQUEST_TIMEOUT_CAP_SECONDS,
            "but an unbounded wait is still refused",
        )

    def test_the_budget_is_capped_by_the_context_however_it_is_configured(self):
        """A budget bigger than the model can read does not make a bigger
        summary. It makes a SHEARED one, and the shear is silent server-side."""
        import ai_summary

        with _settings(ai_local_context_tokens=12288, ai_local_evidence_budget_chars=500_000):
            ceiling = ai_summary.local_evidence_budget_ceiling_chars()
            self.assertEqual(ai_summary.local_evidence_budget_chars(), ceiling)
            self.assertLess(ceiling, 500_000)

    def test_raising_the_context_raises_the_ceiling_proportionally(self):
        import ai_summary

        with _settings(ai_local_context_tokens=12288):
            small = ai_summary.local_evidence_budget_ceiling_chars()
        with _settings(ai_local_context_tokens=65536):
            large = ai_summary.local_evidence_budget_ceiling_chars()
        self.assertGreater(large, small * 4, "a 5x context must buy far more evidence")

    def test_a_configured_budget_under_the_ceiling_is_honoured(self):
        import ai_summary

        with _settings(ai_local_context_tokens=65536, ai_local_evidence_budget_chars=48_000):
            self.assertEqual(ai_summary.local_evidence_budget_chars(), 48_000)

    def test_the_two_chars_per_token_constants_lean_opposite_ways(self):
        """They look interchangeable and are not; merging them reintroduces the
        2026-08-27 shear. Sizing a budget safely assumes text tokenizes BADLY
        (small ratio, small budget); estimating what was sent safely assumes it
        tokenizes WELL (large ratio, small estimate, no false alarm)."""
        import ai_summary

        self.assertLess(
            ai_summary._BUDGET_CHARS_PER_TOKEN,
            ai_summary._ESTIMATED_CHARS_PER_TOKEN,
            "the budget constant must be the pessimistic one",
        )
        # Both must stay anchored to what was actually measured (2.06-2.23).
        self.assertLessEqual(ai_summary._BUDGET_CHARS_PER_TOKEN, 2.06)
        self.assertLessEqual(ai_summary._ESTIMATED_CHARS_PER_TOKEN, 3.0)


class PromptTruncationTripwireTests(unittest.TestCase):
    """Output built from a sheared prompt validates, so it must not be parsed.

    This is the failure that ran for six nights: 80,000 chars of evidence into
    a 2048-token context produced confident JSON about evidence the model never
    saw, and only died because the JSON itself was cut mid-string.
    """

    def setUp(self):
        import tempfile

        import ai_summary

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        with _settings():
            self.evidence = ai_summary.build_evidence_package(
                ["daily_report"], source_overrides=_daily_overrides(Path(self._tmp.name))
            )
        self.summary_text = json.dumps(_valid_summary("daily.auto_report"))

    def _run(self, usage):
        import ai_summary

        calls = []

        def fake_post(url, **kwargs):
            calls.append(kwargs)
            return _chat_response(self.summary_text, usage=usage)

        with _settings(ai_local_endpoint_url=ENDPOINT, ai_local_model_medium="gemma3:12b"):
            result = ai_summary.request_ai_summary(
                provider="local",
                model="",
                api_key="",
                evidence=self.evidence,
                post=fake_post,
            )
        return result, calls

    def test_a_sheared_prompt_raises_instead_of_being_parsed(self):
        with self.assertRaises(RuntimeError) as caught:
            self._run({"prompt_tokens": 12, "completion_tokens": 100})
        message = str(caught.exception)
        self.assertIn("truncated the prompt", message)
        # Both numbers named: a bare "truncated" tells the operator nothing
        # about which side to change.
        self.assertIn("server reported seeing 12", message)
        self.assertIn("ai_local_evidence_budget_chars", message)

    def test_a_sheared_prompt_is_not_retried_into_a_valid_looking_answer(self):
        # The retry sends MORE text, so it would truncate harder. One call only.
        with self.assertRaises(RuntimeError):
            _result, calls = self._run({"prompt_tokens": 5})
        # assertRaises swallowed the return, so re-run counting calls directly.
        import ai_summary

        calls = []

        def fake_post(url, **kwargs):
            calls.append(kwargs)
            return _chat_response(self.summary_text, usage={"prompt_tokens": 5})

        with _settings(ai_local_endpoint_url=ENDPOINT):
            with self.assertRaises(RuntimeError):
                ai_summary.request_ai_summary(
                    provider="local",
                    model="",
                    api_key="",
                    evidence=self.evidence,
                    post=fake_post,
                )
        self.assertEqual(len(calls), 1)

    def test_absent_usage_is_not_treated_as_truncation(self):
        # Some llama.cpp builds omit usage; a missing field is not evidence.
        result, _calls = self._run(None)
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["usage"], {})

    def test_healthy_usage_passes_and_is_recorded(self):
        import ai_summary

        sent = len(ai_summary._local_user_prompt(self.evidence)) + len(
            ai_summary._system_instruction()
        )
        honest = int(sent / 3.5)
        result, _calls = self._run({"prompt_tokens": honest, "completion_tokens": 220})
        self.assertEqual(result["status"], "validated")
        self.assertEqual(
            result["usage"], {"prompt_tokens": honest, "completion_tokens": 220}
        )

    def test_malformed_usage_never_crashes_the_success_path(self):
        for broken in ({"prompt_tokens": "many"}, {"prompt_tokens": True}, {"prompt_tokens": 0}, {}):
            with self.subTest(usage=broken):
                result, _calls = self._run(broken)
                self.assertEqual(result["status"], "validated")


class CloudUsageRecordingTests(unittest.TestCase):
    def test_cloud_usage_is_normalized_from_input_output_token_names(self):
        import ai_summary

        # OpenAI/Anthropic say input/output; the local server says
        # prompt/completion. The ledger should not have to know which.
        self.assertEqual(
            ai_summary.usage_from_body({"usage": {"input_tokens": 10, "output_tokens": 3}}),
            {"prompt_tokens": 10, "completion_tokens": 3},
        )
        self.assertEqual(
            ai_summary.usage_from_body({"usage": {"prompt_tokens": 7, "completion_tokens": 2}}),
            {"prompt_tokens": 7, "completion_tokens": 2},
        )
        self.assertEqual(ai_summary.usage_from_body({}), {})
        self.assertEqual(ai_summary.usage_from_body({"usage": "nope"}), {})


if __name__ == "__main__":
    unittest.main()
