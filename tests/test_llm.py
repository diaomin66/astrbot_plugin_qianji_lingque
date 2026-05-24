from __future__ import annotations

import sys
import unittest
from pathlib import Path

SDK_PATH = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"
if SDK_PATH.exists():
    sys.path.insert(0, str(SDK_PATH))

from qianji_lingque.llm import extract_completion_text, sanitize_reply

try:
    from astrbot.api.provider import LLMResponse as AstrBotLLMResponse
    from astrbot.core.message.message_event_result import MessageChain
except Exception:
    AstrBotLLMResponse = None
    MessageChain = None


class FakeErrorResponse:
    role = "err"
    completion_text = "provider failed"


class LLMParsingTests(unittest.TestCase):
    def test_sanitize_reply_removes_wrapping(self) -> None:
        self.assertEqual(sanitize_reply('"可以，我看看。"\n'), "可以，我看看。")

    def test_sanitize_reply_extracts_safe_json_reply_field(self) -> None:
        self.assertEqual(sanitize_reply('{"reply":"可以，我看看。"}'), "可以，我看看。")

    def test_sanitize_reply_drops_tool_like_output(self) -> None:
        self.assertEqual(sanitize_reply("[TOOL] call done"), "")

    def test_extract_completion_text_hides_provider_errors(self) -> None:
        self.assertEqual(extract_completion_text(FakeErrorResponse()), "")

    def test_extract_completion_text_ignores_unknown_objects(self) -> None:
        self.assertEqual(extract_completion_text(object()), "")

    def test_extract_completion_text_reads_result_chain(self) -> None:
        class Plain:
            def __init__(self, text: str) -> None:
                self.text = text

        class Chain:
            chain = [Plain("链路回复")]

        class Response:
            role = "assistant"
            completion_text = ""
            result_chain = Chain()

        self.assertEqual(extract_completion_text(Response()), "链路回复")

    def test_extract_completion_text_reads_real_astrbot_result_chain(self) -> None:
        if AstrBotLLMResponse is None or MessageChain is None:
            self.skipTest("AstrBot SDK is not available")
        response = AstrBotLLMResponse(
            role="assistant",
            result_chain=MessageChain().message("SDK 链路回复"),
        )

        self.assertEqual(extract_completion_text(response), "SDK 链路回复")


if __name__ == "__main__":
    unittest.main()
