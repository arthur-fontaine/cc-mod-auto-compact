import asyncio
import os
import sys
import types
import unittest
from unittest import mock

from helpers import REPO, CleanEnv

import summarizer


def fake_sdk(seen):
    """A stand-in for claude_agent_sdk: records the options, answers one line, no network."""
    sdk = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content, error=None):
            self.content, self.error = content, error

    class ResultMessage:
        def __init__(self, usage, is_error=False, result=None):
            self.usage, self.total_cost_usd, self.is_error, self.result, self.subtype = usage, 0.0, is_error, result, "x"

    async def query(prompt, options):
        seen.append({"prompt": prompt, "options": options, "environ": dict(os.environ)})
        if "LIMIT" in prompt:
            yield AssistantMessage([TextBlock("You've hit your usage limit · resets 3pm")], error="rate_limit")
            yield ResultMessage({}, is_error=True, result="usage limit")
            return
        yield AssistantMessage([TextBlock("task: t [done]")])
        yield ResultMessage({"input_tokens": 10, "output_tokens": 5})

    sdk.ClaudeAgentOptions, sdk.TextBlock, sdk.AssistantMessage, sdk.ResultMessage = (
        ClaudeAgentOptions, TextBlock, AssistantMessage, ResultMessage)
    sdk.query = query
    return sdk


class IsolationTest(unittest.TestCase):
    def call(self, prompt="<input>\nuser: hi\n</input>", **env):
        seen = []
        with CleanEnv(ANTHROPIC_API_KEY="sk-test-api-key", ANTHROPIC_AUTH_TOKEN="tok-test", **env), \
                mock.patch.dict(sys.modules, {"claude_agent_sdk": fake_sdk(seen)}):
            result = asyncio.run(summarizer.complete(prompt))
            mode = summarizer.auth_mode()
        return seen[0], result, mode

    def test_options_isolate_the_call(self):
        seen, (text, usage), mode = self.call()
        o = seen["options"]
        self.assertEqual(text, "task: t [done]")
        self.assertEqual(o.setting_sources, [])
        self.assertEqual(o.mcp_servers, {})
        self.assertTrue(o.strict_mcp_config)
        self.assertEqual(o.tools, [])
        self.assertEqual(o.allowed_tools, [])
        self.assertEqual(o.plugins, [])
        self.assertEqual(o.max_turns, 1)
        self.assertIn("safe-mode", o.extra_args)
        self.assertEqual(o.model, "claude-haiku-5-5")
        self.assertEqual(o.system_prompt, (REPO / "prompts" / "compact.md").read_text())

    def test_api_credentials_are_stripped_by_default(self):
        seen, _, mode = self.call()
        self.assertEqual(mode, "subscription")
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            self.assertNotIn(var, seen["environ"])
            self.assertNotIn(var, seen["options"].env)

    def test_api_credentials_are_kept_only_with_summarizer_use_api(self):
        seen, _, mode = self.call(SUMMARIZER_USE_API="1")
        self.assertEqual(mode, "API")
        self.assertEqual(seen["environ"]["ANTHROPIC_API_KEY"], "sk-test-api-key")

    def test_the_haiku_alias_is_refused(self):
        with CleanEnv(SUMMARIZER_MODEL="haiku"):
            with self.assertRaises(SystemExit):
                summarizer.settings()
        with CleanEnv(SUMMARIZER_MODEL="claude-haiku-5-5"):
            self.assertEqual(summarizer.settings()["model"], "claude-haiku-5-5")

    def test_a_usage_limit_stops_with_the_reset_time(self):
        with self.assertRaises(summarizer.UsageLimit) as ctx:
            self.call(prompt="LIMIT")
        self.assertIn("3pm", ctx.exception.reset)
        block = summarizer.usage_stop_block(ctx.exception, "resume-cmd")
        self.assertTrue(block.startswith("CLAUDE USAGE STOP"))
        self.assertIn("resume-cmd", block)


if __name__ == "__main__":
    unittest.main()
