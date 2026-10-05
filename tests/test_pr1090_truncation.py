"""Regressions for the max_tokens truncation detection merged from PR #1090.

This is the highest-impact behaviour change in that batch: a response cut off by
max_tokens used to be written out as a finished translation, producing half
sentences. It now raises so the caller retries.
"""
import unittest
from types import SimpleNamespace

from ModuleFolders.Infrastructure.LLMRequester.OpenaiRequester import OpenaiRequester


def _response(content="译文", reasoning=None, finish_reason="stop", model="m"):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model=model,
        usage=None,
    )


class TruncationDetectionTests(unittest.TestCase):
    def setUp(self):
        self.requester = OpenaiRequester()

    def test_truncated_completion_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            self.requester._extract_from_completion(_response(finish_reason="length"))
        self.assertIn("length", str(caught.exception))

    def test_complete_response_still_extracts(self):
        think, content, _, _ = self.requester._extract_from_completion(_response())
        self.assertEqual(content, "译文")
        self.assertEqual(think, "")

    def test_reasoning_still_extracted_when_not_truncated(self):
        """Interaction check: our OpenRouter reasoning work must survive the merge."""
        think, content, _, _ = self.requester._extract_from_completion(
            _response(content="译文", reasoning="先想想")
        )
        self.assertEqual(think, "先想想")
        self.assertEqual(content, "译文")

    def test_truncation_wins_over_reasoning_extraction(self):
        with self.assertRaises(RuntimeError):
            self.requester._extract_from_completion(
                _response(content="译文", reasoning="先想想", finish_reason="length")
            )

    def test_truncated_sse_stream_raises(self):
        raw = (
            'data: {"choices":[{"delta":{"content":"半句"}}]}\n'
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n'
        )
        with self.assertRaises(RuntimeError):
            self.requester._parse_sse_response(raw)

    def test_complete_sse_stream_still_parses(self):
        raw = (
            'data: {"choices":[{"delta":{"content":"译"}}]}\n'
            'data: {"choices":[{"delta":{"content":"文"}}]}\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":7,"completion_tokens":3}}\n'
        )
        think, content, prompt_tokens, completion_tokens = self.requester._parse_sse_response(raw)
        self.assertEqual(content, "译文")
        self.assertEqual(prompt_tokens, 7)
        self.assertEqual(completion_tokens, 3)

    def test_openrouter_sse_reasoning_survives(self):
        raw = (
            'data: {"choices":[{"delta":{"reasoning":"想一"}}]}\n'
            'data: {"choices":[{"delta":{"reasoning":"想二"}}]}\n'
            'data: {"choices":[{"delta":{"content":"译文"}}]}\n'
        )
        think, content, _, _ = self.requester._parse_sse_response(raw)
        self.assertEqual(think, "想一想二")
        self.assertEqual(content, "译文")


if __name__ == "__main__":
    unittest.main()
