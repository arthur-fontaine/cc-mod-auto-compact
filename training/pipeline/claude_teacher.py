"""A Claude model as the decision teacher: the student's own prompt, one letter back.

    uv run python pipeline/evaluate.py --claude claude-haiku-5-5 --split dev --name haiku-v1-dev

The teacher reads exactly what the student is trained on (`task.messages` or `messages_v2`:
the same system prompt and JSON task) and answers with one option letter, a hard label as in
Nisev and tev1. Runs through `claude -p` isolated like the hindsight teachers (hindsight.env),
on the subscription. Answers are cached by model and prompt, so a rerun asks nothing again;
CLAUDE_TEACHER_CACHE points at another cache, to draw the same points a second time.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402
from hindsight import env  # noqa: E402
from task import CATEGORY_OF, LETTERS, messages, messages_v2  # noqa: E402

CACHE = Path(os.environ.get("CLAUDE_TEACHER_CACHE", DATA / "claude-cache"))
TRIES = 3


def prompt_of(state):
    return (messages_v2 if "memory" in state else messages)(state)


def letter(text):
    text = text.strip().strip("`*\"' ").upper()
    return text[0] if text[:1] and text[0] in LETTERS and (len(text) == 1 or not text[1].isalpha()) else None


def call(model, system, user):
    with tempfile.TemporaryDirectory() as cwd:
        done = subprocess.run(
            ["claude", "-p", "--model", model, "--safe-mode", "--setting-sources", "", "--strict-mcp-config",
             "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--no-session-persistence",
             "--tools", "", "--output-format", "json", "--system-prompt", system],
            input=user, capture_output=True, text=True, cwd=cwd, timeout=300, env=env())
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout)[-400:])
    return json.loads(done.stdout)["result"]


def ask(model, state):
    """The teacher's category for a state, and the call's latency (0 when cached)."""
    system, user = (m["content"] for m in prompt_of(state))
    key = hashlib.sha256(json.dumps([model, system, user]).encode()).hexdigest()
    path = CACHE / model / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["category"], 0.0
    started = time.perf_counter()
    for _ in range(TRIES):
        reply = call(model, system, user)
        if (choice := letter(reply)) is not None:
            break
    else:
        raise RuntimeError(f"no option letter in {reply[:80]!r}")
    ms = (time.perf_counter() - started) * 1000
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"category": CATEGORY_OF[choice], "reply": reply[:200]}))
    return CATEGORY_OF[choice], ms
