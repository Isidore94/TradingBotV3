import unittest
from unittest.mock import patch

from market_prep.models import MarketPrepConfig
from market_prep.report_builder import attach_ai_summary_to_report
from market_prep.services import llm_summary_service


class FakeResponse:
    def __init__(self, *, status_code=200, text="", payload=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload if payload is not None else {}
        self.headers = headers or {"content-type": "text/html"}

    def json(self):
        return self._payload


class MarketPrepLlmSummaryServiceTests(unittest.TestCase):
    def _report(self):
        return {
            "report_type": "daily",
            "report_date": "2026-05-05",
            "generated_at": "2026-05-05T06:00:00",
            "trading_posture": ["Reduce size around scheduled macro event."],
            "catalyst_clock": [
                {
                    "date": "2026-05-05",
                    "time_et": "08:30",
                    "bucket": "Macro",
                    "priority": "HIGH",
                    "text": "CPI",
                }
            ],
            "todays_events": {
                "events": [
                    {
                        "date": "2026-05-05",
                        "time_et": "08:30",
                        "priority": "HIGH",
                        "currency": "USD",
                        "event": "CPI",
                    }
                ]
            },
            "rss_headlines": {
                "headlines": [
                    {
                        "title": "Treasury yields rise before CPI",
                        "source": "Test News",
                        "url": "https://example.com/cpi",
                        "query": "Treasury auction yields",
                        "summary": "Bond market braces for inflation data.",
                        "tags": ["Fed/rates", "inflation"],
                    },
                    {
                        "title": "Lifestyle story",
                        "source": "Other",
                        "url": "https://example.com/lifestyle",
                        "tags": [],
                    },
                ]
            },
        }

    def test_generate_summary_fetches_capped_articles_and_calls_openai(self):
        config = MarketPrepConfig(
            llm_summary={
                "model": "gpt-5-mini",
                "max_output_tokens": 220,
                "headline_limit": 10,
                "article_limit": 1,
                "article_char_limit": 900,
                "request_timeout_seconds": 12,
                "article_timeout_seconds": 5,
            }
        )
        article_html = """
        <html><body><article>
        <p>Treasury yields moved higher as traders prepared for CPI and the next auction cycle.</p>
        <p>Investors focused on whether inflation would keep the Federal Reserve cautious.</p>
        </article></body></html>
        """

        def fake_post(_url, *, headers, json, timeout):
            self.assertEqual(headers["Authorization"], "Bearer sk-test")
            self.assertEqual(json["model"], "gpt-5-mini")
            self.assertEqual(json["max_output_tokens"], 220)
            self.assertIn("Article snippets", json["input"])
            self.assertIn("Treasury yields rise before CPI", json["input"])
            self.assertEqual(timeout, 12)
            return FakeResponse(payload={"output_text": "Macro risk is CPI/rates-led. Used: Test News."})

        with patch.object(llm_summary_service.requests, "get", return_value=FakeResponse(text=article_html)) as get_mock:
            with patch.object(llm_summary_service.requests, "post", side_effect=fake_post) as post_mock:
                result = llm_summary_service.generate_market_prep_llm_summary(
                    self._report(),
                    config=config,
                    api_key="sk-test",
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["article_count"], 1)
        self.assertIn("CPI/rates-led", result["summary"])
        get_mock.assert_called_once()
        post_mock.assert_called_once()

    def test_missing_key_skips_request(self):
        with patch.object(llm_summary_service, "get_market_prep_openai_api_key", return_value=""):
            with patch.object(llm_summary_service.requests, "post") as post_mock:
                result = llm_summary_service.generate_market_prep_llm_summary(
                    self._report(),
                    config=MarketPrepConfig(),
                )

        self.assertEqual(result["status"], "missing_key")
        post_mock.assert_not_called()

    def test_attach_ai_summary_to_daily_report_adds_markdown_section(self):
        report = {
            "report_type": "daily",
            "report_date": "2026-05-05",
            "scheduled_landmines": {},
            "todays_events": {"events": []},
            "next_7_events": {"events": []},
            "today_tomorrow_earnings": {"earnings": []},
            "next_7_earnings": {"earnings": []},
            "watchlist_risk": {"risks": []},
            "rss_headlines": {"headlines": []},
            "youtube_links": {"videos": []},
            "trading_posture": [],
        }

        updated = attach_ai_summary_to_report(
            report,
            {
                "status": "ok",
                "status_label": "Ready",
                "model": "gpt-5-mini",
                "summary": "Brief macro summary.",
                "used_articles": [{"title": "CPI preview"}],
            },
        )

        self.assertIn("## AI Macro Brief", updated["markdown"])
        self.assertIn("Brief macro summary.", updated["markdown"])


if __name__ == "__main__":
    unittest.main()
