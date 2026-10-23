"""Text rules shared by the pipeline, the training packer and the mod (hooks/text.js).

Everything here has a JS twin; tests/fixtures.js holds cases generated from this file
and `claude plugin test` checks the JS against them, so training and serving clip,
group and render a session the same way.

- `clip` keeps both ends of a text within a byte budget and a token budget.
- `estimate_tokens` is a deterministic upper estimate of Qwen3.5 tokens (Mammoth is a
  Qwen3.5-27B), checked against the real tokenizer in tests/test_text.py.
- `messages` turns transcript rows ($.session.messages() shape: role, text, toolUses)
  into the tree's messages: each human prompt, and everything the agent did between two
  prompts as one `agent` message. Tool results are dropped.
"""
import json
import math
import re

MARKER = "\n[… truncated …]\n"
BYTES_PER_TOKEN = 3  # Qwen averages ~4 bytes per token on English and code; 3 leaves margin.
MAX_AGENT_BYTES = 16_000  # one agent message, before Haiku compresses it
MAX_TOOL_INPUT_BYTES = 300

SKIP_PREFIXES = (
    "<local-command-", "<command-name>", "<command-message>", "<bash-input>", "<bash-stdout>",
    "<bash-stderr>", "<task-notification>", "[Request interrupted", "Caveat:", "<system-reminder>",
    "This session is being continued", "[REDACTED_CONFIG_DUMP", "<ci_monitor_event>", "<ci-monitor-event>",
    # Some exporters write subagent and result events as user text.
    "[task_started]", "[task_notification]", "[task_progress]", "[init]", "[result]",
)


def utf8_len(text):
    return len(text.encode("utf-8"))


def estimate_tokens(text):
    return math.ceil(utf8_len(text) / BYTES_PER_TOKEN)


def _head(chars, limit):
    out, size = [], 0
    for c in chars:
        n = utf8_len(c)
        if size + n > limit:
            break
        out.append(c)
        size += n
    return "".join(out)


def clip(text, max_bytes, max_tokens=None):
    """`text` cut in the middle to fit both budgets, keeping its start and its end."""
    text = text or ""
    limit = max_bytes if max_tokens is None else min(max_bytes, max_tokens * BYTES_PER_TOKEN)
    if utf8_len(text) <= limit:
        return text
    half = max(0, (limit - utf8_len(MARKER)) // 2)
    chars = list(text)
    head = _head(chars, half)
    tail = _head(reversed(chars), half)[::-1]
    return head + MARKER + tail


def clean_prompt(text):
    """What the user typed, with exporter and IDE envelopes removed; None if nothing human."""
    text = (text or "").strip()
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.S)
    if m:
        text = m.group(1)
    text = re.sub(r"<timestamp>.*?</timestamp>", "", text, flags=re.S)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(r"<image_files>.*?</image_files>", "", text, flags=re.S)
    text = re.sub(r"<ide_(opened_file|selection|diagnostics)>.*?</ide_\1>", "", text, flags=re.S)
    text = re.sub(r"^(\[Image[^\]]*\]\s*)+", "", text)
    text = text.strip()
    if not text or text.startswith(SKIP_PREFIXES):
        return None
    return text


def command_of(text):
    """`clear` or `compact` when the row is that slash command, else None."""
    m = re.search(r"<command-name>/(clear|compact)</command-name>", text or "")
    return m.group(1) if m else None


def tool_line(use):
    args = json.dumps(use.get("input") or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    line = f"tool: {use.get('tool') or '?'} {clip(args, MAX_TOOL_INPUT_BYTES)}"
    return line + (" (error)" if use.get("isError") else "")


def messages(rows):
    """Tree messages from transcript rows: [{kind: user|agent, text, row}], oldest first.

    `row` is the index of the row that starts the message. User rows that only carry
    tool results, and rows that are not a human prompt (commands, meta), are skipped.
    """
    out = []
    for i, r in enumerate(rows):
        if r.get("role") == "user":
            if r.get("toolResults"):
                continue
            prompt = clean_prompt(r.get("text"))
            if prompt is not None:
                out.append({"kind": "user", "text": prompt, "row": i})
            continue
        parts = []
        if (r.get("text") or "").strip():
            parts.append("agent: " + r["text"].strip())
        parts += [tool_line(u) for u in r.get("toolUses") or []]
        if not parts:
            continue
        if out and out[-1]["kind"] == "agent":
            out[-1]["text"] += "\n" + "\n".join(parts)
        else:
            out.append({"kind": "agent", "text": "\n".join(parts), "row": i})
    return out


def render(message):
    """A message as a summarizer reads it, and as a leaf when it fits in 512 bytes."""
    if message["kind"] == "user":
        return clip("user: " + message["text"], MAX_AGENT_BYTES)
    return clip(message["text"], MAX_AGENT_BYTES)
