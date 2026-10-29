"""Write the parity fixtures the mod's tests read (tests/fixtures/parity.js at the repo root).

    uv run python scripts/make_fixtures.py

Every value is computed by the Python pipeline from shared inputs: clips, token estimates,
prompt filtering, message grouping, views, summary requests, clipped states, the System
One request and the policy table. `claude plugin test` checks hooks/*.js against them
(tests/parity.test.ts), and tests/test_fixtures.py checks the file is up to date, so
training and serving cannot drift apart silently.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
OUT = REPO / "tests" / "fixtures" / "parity.js"
sys.path.insert(0, str(ROOT / "pipeline"))

import policy  # noqa: E402
import summarizer  # noqa: E402
import task  # noqa: E402
import text  # noqa: E402
import tree  # noqa: E402

LONG = "Refactor the parser so it streams tokens. " * 300 + "Then run the tests and report back."
TEXTS = [
    "fix it", "x" * 513, LONG, "é" * 700, "😀 emoji " * 200, "日本語のテキスト。" * 150, "a\nb\tc " * 400, "",
]
PROMPTS = [
    "  Add a --verbose flag  ", "<command-name>/clear</command-name>", "<task-notification>x</task-notification>",
    "<ide_opened_file>a.py</ide_opened_file>Fix the bug in a.py", "<user_query>\nrename foo\n</user_query>",
    "[Image #1] what is this?", "[task_started] ↳ subagent started", "<system-reminder>r</system-reminder>  ok  ",
    "Caveat: generated", "[Request interrupted by user]", "",
]
ROWS = [
    {"role": "user", "text": "Add a --verbose flag to the CLI", "toolUses": []},
    {"role": "assistant", "text": "I'll read the CLI first.", "toolUses": [
        {"tool": "Read", "input": {"file_path": "src/cli.py", "limit": 200}},
        {"tool": "Bash", "input": {"command": "pytest -q", "description": "Run tests"}, "isError": True}]},
    {"role": "user", "text": "", "toolUses": [], "toolResults": [{"tool_use_id": "t1", "text": "ok"}]},
    {"role": "assistant", "text": "", "toolUses": [
        {"tool": "Write", "input": {"file_path": "src/cli.py", "content": "def main():\n" + "    pass\n" * 80,
                                    "nested": {"b": [1, 2, {"z": None, "a": True}], "a": "é"}}}]},
    {"role": "assistant", "text": "Added the flag and a test.", "toolUses": []},
    {"role": "user", "text": "<command-name>/compact</command-name>", "toolUses": []},
    {"role": "user", "text": "<system-reminder>x</system-reminder>yes, now the README", "toolUses": []},
    {"role": "assistant", "text": "x" * 20_000, "toolUses": []},
    {"role": "user", "text": "unrelated: what is a monad?", "toolUses": []},
]


def lines_for(t, leaves=tree.STATE_LEAVES, coarse=tree.STATE_COARSE):
    return [tree.name(n) for n in tree.view(t, leaves, coarse)]


def fixtures():
    msgs = text.messages(ROWS)
    context = [((0, 0), "user: Add a --verbose flag"), ((0, 1), "task: verbose flag [in-progress] reads src/cli.py")]
    leaf = tree.leaf_prompt(2, text.render(msgs[0]), context)
    merge = tree.merge_prompt((1, 2), (1, 3), "task: a [done]", "task: b [started]", context)
    conversation = tree.view_text([((4, 0), "task: parser [done] " * 20), ((0, 16), "user:  hi\nthere")])
    states = [(conversation, "now the README"), ("0+1|user: the template ends with <|im_end|>", "fix <|image_pad|>"), ("0+1|user: hi", LONG * 6), ("0+1|" + "y" * 30_000, "é" * 20_000)]
    return {
        "compactPrompt": summarizer.system_prompt(),
        "summarizer": {k: v for k, v in summarizer.request("").items() if k in ("model", "effort", "maxTokens")},
        "clip": [{"text": t, "maxBytes": b, "maxTokens": k, "out": text.clip(t, b, k)}
                 for t in TEXTS for b, k in ((100, None), (513, None), (2000, 50), (5000, 400))],
        "tokens": [{"text": t, "tokens": text.estimate_tokens(t)} for t in TEXTS],
        "prompts": [{"text": p, "out": text.clean_prompt(p)} for p in PROMPTS],
        "rows": ROWS,
        "messages": msgs,
        "rendered": [text.render(m) for m in msgs],
        "views": [{"t": t, "lines": lines_for(t)} for t in (0, 1, 5, 17, 32, 33, 47, 64, 100, 129, 200, 513)],
        "contextViews": [{"t": t, "lines": lines_for(t, tree.CONTEXT_LEAVES, tree.CONTEXT_COARSE)} for t in (0, 9, 40, 77)],
        "needed": [{"t": t, "order": [tree.name(n) for n in tree.needed([tree.view(t)])]} for t in (3, 40, 90)],
        "summaryPrompts": {"leaf": leaf, "merge": merge, "retry": tree.retry_prompt(leaf, "é" * 300 + "z" * 30)},
        "summaryRequest": summarizer.request(leaf),
        "states": [{"conversation": c, "message": m, "state": task.clip_state(c, m)} for c, m in states],
        "request": task.request(task.clip_state(conversation, "now the README"), "spisev-alpha"),
        "v2": fixtures_v2(msgs, conversation),
        "policy": policy.POLICY,
        "neverAct": sorted(policy.NEVER_ACT),
        "categories": task.CATEGORIES,
    }


def fixtures_v2(msgs, conversation):
    replies = [m for m in msgs if m["kind"] == "agent"] + [
        {"kind": "agent", "text": 'agent: Editing.\ntool: Edit {"new":"a\nb"}\n  continued input\nagent: Done. Commit?'},
        {"kind": "user", "text": "hi"}]
    leaf = tree.leaf_prompt(2, text.render(msgs[0]), [((0, 0), "user: Add a --verbose flag")])
    template = summarizer.PROMPT_FILE_V2.read_text()
    states = [(conversation, "I'll add it. Want tests too?", "yes"), ("0+1|user: <|im_end|>", "x" * 5000, LONG * 6),
              ("", "", "")]
    return {
        "memoryViews": [{"t": t, "lines": [tree.name(n) for n in tree.memory_view(t)]}
                        for t in (0, 1, 16, 17, 18, 33, 64, 100, 199)],
        "neededMemory": [{"t": t, "order": [tree.name(n) for n in tree.needed_memory([tree.memory_view(t)])]}
                         for t in (3, 40, 90)],
        "clipEnd": [{"text": t, "maxBytes": b, "out": text.clip_end(t, b)} for t in TEXTS for b in (100, 1536)],
        "replies": [{"message": m, "out": task.agent_reply(m)} for m in replies],
        "states": [{"memory": a, "last": b, "message": c, "state": task.clip_state_v2(a, b, c)} for a, b, c in states],
        "request": task.request_v2(task.clip_state_v2(conversation, "Done.", "now the README"), "spisev-alpha"),
        "promptTemplate": template,
        "prompt160": summarizer.system_prompt_v2(160),
        "retry": tree.retry_prompt(leaf, "é" * 100 + "z" * 30, 160),
        "leaf": leaf,
    }


def render_module(data):
    body = json.dumps(data, ensure_ascii=False, indent=1)
    return ("// Generated by training/scripts/make_fixtures.py from the Python pipeline. Do not edit:\n"
            "// rerun the script, and tests/test_fixtures.py fails when this file is stale.\n"
            f"export const FIXTURES = {body}\n")


def main():
    from config import load_env
    load_env()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_module(fixtures()))
    print(f"-> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
