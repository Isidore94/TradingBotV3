"""ntfy push channel: request construction and fail-quiet delivery."""

import sys
import urllib.error
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import push_notify  # noqa: E402


CONFIG = {"server": "https://ntfy.example.com/", "topic": "trader-topic", "token": ""}


def test_unconfigured_topic_is_a_silent_noop():
    assert push_notify.push_configured({"topic": ""}) is False
    assert push_notify.build_push_request("t", "m", config={"topic": ""}) is None
    result = push_notify.send_push("t", "m", config={"topic": ""})
    assert result["ok"] is False and result["error"] == ""
    # Nothing was transmitted, which is why a caller may retry immediately -
    # there is no chance a notification is already on its way (R2.1).
    assert result["kind"] == "unconfigured"


def test_request_carries_topic_priority_and_body():
    request = push_notify.build_push_request(
        "Price alert", "SPY crossed", config=CONFIG, priority="urgent", tags="rotating_light"
    )
    assert request.full_url == "https://ntfy.example.com/trader-topic"
    assert request.get_method() == "POST"
    assert request.data == b"SPY crossed"
    assert request.get_header("Title") == "Price alert"
    assert request.get_header("Priority") == "urgent"
    assert request.get_header("Tags") == "rotating_light"
    assert request.get_header("Authorization") is None


def test_bad_priority_falls_back_to_high_and_token_adds_bearer():
    request = push_notify.build_push_request(
        "t", "m", config={**CONFIG, "token": "tk_secret"}, priority="loudest"
    )
    assert request.get_header("Priority") == "high"
    assert request.get_header("Authorization") == "Bearer tk_secret"


def test_send_push_success_and_failure_never_raise():
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    sent = []

    def opener(request, timeout):
        sent.append(request)
        return _Response()

    delivered = push_notify.send_push("t", "m", config=CONFIG, opener=opener)
    assert delivered["ok"] is True and delivered["error"] == ""
    assert delivered["kind"] == "delivered"
    assert len(sent) == 1

    def broken_opener(request, timeout):
        raise urllib.error.URLError("dns down")

    result = push_notify.send_push("t", "m", config=CONFIG, opener=broken_opener)
    assert result["ok"] is False and "dns down" in result["error"]
    # The request was already on the wire, so whether it arrived is genuinely
    # unknown - a different problem from the server saying no, and the reason
    # a retrying caller backs off instead of resending.
    assert result["kind"] == "ambiguous"
