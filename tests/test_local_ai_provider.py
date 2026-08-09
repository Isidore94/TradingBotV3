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

    sections = {name: [] for name in ai_summary.AI_SUMMARY_SECTIONS}
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


def _chat_response(text: str, *, status_code: int = 200):
    return _Response(
        {"id": "chatcmpl-local", "choices": [{"message": {"role": "assistant", "content": text}}]},
        status_code=status_code,
    )


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
        for key in ("executive_summary", *ai_summary.AI_SUMMARY_SECTIONS):
            self.assertIn(key, user_message)
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


if __name__ == "__main__":
    unittest.main()
