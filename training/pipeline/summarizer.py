"""The only Python code path that calls Haiku: one tool-less completion per summary node.

    uv run python pipeline/summarizer.py --check      # print the auth mode and the init check, one call

Runs through the Claude Agent SDK with the Claude Code login of whoever runs it (a Claude
subscription), isolated from that person's setup, every option passed explicitly:
no filesystem setting sources (so no CLAUDE.md, settings, hooks or plugins), safe mode
(no installed plugins or mods, skills-dir mods included), no MCP servers, no tools, no
slash commands, no session persistence, one turn. The system prompt is
prompts/compact.md, versioned by its hash. The model is pinned by full id, never the
`haiku` alias, which follows ANTHROPIC_DEFAULT_HAIKU_MODEL.

ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are removed from the SDK's environment (and
this process's) unless SUMMARIZER_USE_API=1, so nothing silently switches to API billing.

The request is built exactly as the mod's $.model.complete call (hooks/summarizer.js):
system = prompts/compact.md, one user message, the same effort. tests/fixtures.js
holds the shared cases.
"""
import asyncio
import hashlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPO, load_env  # noqa: E402

PROMPT_FILE = REPO / "prompts" / "compact.md"
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "low"
API_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# List price, USD per million tokens (claude-haiku-5-5, prompts up to 100K tokens).
PRICE = {"claude-haiku-5-5": (0.10, 0.50)}


class UsageLimit(Exception):
    """The Claude usage limit or rate limit was hit: stop every Haiku call."""

    def __init__(self, message, reset=None):
        super().__init__(message)
        self.reset = reset


def system_prompt():
    return PROMPT_FILE.read_text()


def prompt_version():
    return hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()[:12]


def settings():
    model = os.environ.get("SUMMARIZER_MODEL") or DEFAULT_MODEL
    if model in ("haiku", "sonnet", "opus") or not model.startswith("claude-"):
        raise SystemExit(f"SUMMARIZER_MODEL must be a full model id such as {DEFAULT_MODEL}, not {model!r}")
    return {"model": model, "effort": os.environ.get("SUMMARIZER_EFFORT") or DEFAULT_EFFORT}


def use_api():
    return os.environ.get("SUMMARIZER_USE_API") == "1"


def auth_mode():
    return "API" if use_api() and any(os.environ.get(v) for v in API_VARS) else "subscription"


def sdk_env():
    """What the SDK subprocess sees: this process's environment, API credentials stripped."""
    if not use_api():
        for var in API_VARS:
            os.environ.pop(var, None)
    return {"CLAUDE_AGENT_SDK_CLIENT_APP": "spisev-summarizer/0.1"}


def options(cwd):
    from claude_agent_sdk import ClaudeAgentOptions

    s = settings()
    return ClaudeAgentOptions(
        model=s["model"],
        effort=s["effort"],
        system_prompt=system_prompt(),
        setting_sources=[],
        mcp_servers={},
        strict_mcp_config=True,
        tools=[],
        allowed_tools=[],
        plugins=[],
        skills=[],
        max_turns=1,
        cwd=cwd,
        env=sdk_env(),
        extra_args={"safe-mode": None, "disable-slash-commands": None, "no-session-persistence": None},
    )


LIMIT_TEXT = re.compile(r"(?i)usage limit|rate limit|limit reached|out of (extra )?usage|too many requests|"
                        r"overloaded|resets? at|5-hour limit|weekly limit")


def _reset_time(text):
    m = re.search(r"(?i)resets?\s+(at\s+)?([^\n.|]{3,40})", text or "")
    return m.group(2).strip() if m else None


async def complete(prompt):
    """One completion: (text, usage dict). Raises UsageLimit on a usage or rate limit."""
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

    text, usage, error = "", {}, None
    with tempfile.TemporaryDirectory() as cwd:
        async for message in query(prompt=prompt, options=options(cwd)):
            if isinstance(message, AssistantMessage):
                if getattr(message, "error", None):
                    error = str(message.error)
                text += "".join(b.text for b in message.content if isinstance(b, TextBlock))
            elif isinstance(message, ResultMessage):
                usage = dict(message.usage or {})
                usage["total_cost_usd"] = message.total_cost_usd
                if message.is_error:
                    error = error or (message.result or message.subtype)
    if error and (LIMIT_TEXT.search(error) or LIMIT_TEXT.search(text) or "rate_limit" in error):
        raise UsageLimit(error, _reset_time(error + " " + text))
    if error:
        raise RuntimeError(f"summary call failed: {error[:300]}")
    return text, usage


def list_price(usage, model):
    inp, out = PRICE.get(model, PRICE[DEFAULT_MODEL])
    tokens_in = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                  "cache_read_input_tokens"))
    return (tokens_in * inp + (usage.get("output_tokens") or 0) * out) / 1e6


def notify_usage_stop():
    import subprocess

    subprocess.run(["osascript", "-e", 'display notification "Claude usage limit, summaries paused" '
                                       'with title "spisev" sound name "Glass"'], check=False)


def usage_stop_block(error, resume):
    return "\n".join([
        "CLAUDE USAGE STOP",
        f"  reason: {error}",
        f"  reset: {error.reset or 'not given'}",
        "  cache: kept (data/tree-cache/)",
        f"  resume: {resume}",
    ])


async def _check():
    from claude_agent_sdk import SystemMessage, query

    print(f"auth mode: {auth_mode()}", flush=True)
    s = settings()
    print(f"model {s['model']} · effort {s['effort']} · prompt {prompt_version()}", flush=True)
    started = time.time()
    with tempfile.TemporaryDirectory() as cwd:
        async for m in query(prompt="<input>\nuser: hello\n</input>", options=options(cwd)):
            if isinstance(m, SystemMessage) and m.subtype == "init":
                d = m.data
                print("init:", {k: d.get(k) for k in ("model", "tools", "mcp_servers", "slash_commands",
                                                      "plugins", "skills", "agents", "apiKeySource")})
    print(f"one call: {time.time() - started:.1f}s", flush=True)
    text, usage = await complete("<input>\nuser: please rename foo to bar in utils.py\n</input>")
    print("reply:", text)
    print("usage:", {k: usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                "total_cost_usd")})


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    load_env()
    if args.check:
        asyncio.run(_check())


if __name__ == "__main__":
    main()
