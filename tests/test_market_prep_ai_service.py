import os
import unittest
from unittest.mock import patch

from market_prep.models import MarketPrepConfig
from market_prep.services.ai_service import build_market_prep_ai_brief, resolve_openai_api_key


class MarketPrepAiServiceTests(unittest.TestCase):
    def test_resolve_openai_api_key_prefers_environment(self):
        config = MarketPrepConfig(api_keys={"openai": "config-key"})

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            self.assertEqual(resolve_openai_api_key(config), "env-key")

    def test_resolve_openai_api_key_falls_back_to_config(self):
        config = MarketPrepConfig(api_keys={"openai": "config-key"})

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_openai_api_key(config), "config-key")

    def test_ai_brief_returns_fallback_without_key(self):
        config = MarketPrepConfig(market_prep_ai={"enabled": True})

        with patch.dict(os.environ, {}, clear=True):
            payload = build_market_prep_ai_brief({"report_type": "daily", "markdown": "Test"}, config=config)

        self.assertEqual(payload["status"], "missing_key")
        self.assertIn("No OpenAI API key", payload["status_label"])
        self.assertIn("start-of-day landmine checklist", payload["prompt"])


if __name__ == "__main__":
    unittest.main()
