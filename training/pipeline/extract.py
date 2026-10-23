"""Turn raw Claude Code, Pi and Codex session files into sessions of transcript rows and prompts.

    uv run python pipeline/extract.py --sources public          # data/raw/** -> data/sessions.jsonl
    uv run python pipeline/extract.py --sources local           # ~/.claude/projects, read only

Each session keeps its main-thread rows in the shape the mod reads from
$.session.messages() (role, text, toolUses), so the tree is built from what serving
sees. Dropped, as in Nisev: subagent (sidechain) messages, meta records, compaction
summaries, tool results (only their error flag is kept, on the tool use). Local
commands, task notifications, interruption markers and IDE context are left out of
the prompts by text.clean_prompt, the same filter the mod applies.

Per human prompt the record notes:
- `preceded_by`: `clear` or `compact` when that command came right before it;
- `interrupted`, and whether the agent replied;
- hindsight, kept apart and never shown to a model: files read and edited in the turn,
  their overlap with files touched before, earlier identifiers the prompt reuses, and
  later prompts that refer back to what came before it.

Every text is scrubbed of secrets before it is written. A tool use is stored as the line
the tree renders (text.tool_line) from its full input, so the input itself is not kept.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, load_env  # noqa: E402
from scrub import scrub, scrub_value  # noqa: E402
from text import clean_prompt, command_of, tool_line  # noqa: E402

RAW = DATA / "raw"
LOCAL = Path.home() / ".claude" / "projects"
OUT = DATA / "sessions.jsonl"
READ_TOOLS = {"read", "grep", "glob", "ls", "notebookread"}
EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit", "apply_patch"}
IDENT = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*[._/][A-Za-z0-9_./]+|[a-z]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+|"
                   r"[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+)\b")


def block_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def tool_results(content):
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") in ("tool_result", "toolResult")]


def tool_use(name, args, use_id):
    args = args if isinstance(args, dict) else {}
    path = args.get("file_path") or args.get("path") or args.get("notebook_path")
    use = {"tool": name or "?", "id": use_id, "input": scrub_value(args)}
    return {"tool": use["tool"], "id": use_id, "line": tool_line(use),
            "path": scrub(path) if isinstance(path, str) else None}


class Session:
    def __init__(self, source, path, project):
        self.source, self.path, self.project = source, path, project
        self.session = None
        self.rows, self.prompts = [], []
        self.uses = {}
        self.pending_command = None
        self.start = self.end = None

    def stamp(self, ts):
        if ts:
            self.start = self.start or ts
            self.end = ts

    def user_text(self, text, ts, meta=False):
        """A user row with text: a prompt, a command, or something the mod's filter drops."""
        if text.strip().startswith("[Request interrupted"):
            if self.prompts:
                self.prompts[-1]["interrupted"] = True
            return
        cmd = command_of(text)
        if cmd:
            self.pending_command = cmd
        if meta:
            return
        text = scrub(text)
        self.rows.append({"role": "user", "text": text, "toolUses": []})
        if clean_prompt(text) is not None:
            self.prompts.append({"row": len(self.rows) - 1, "preceded_by": self.pending_command,
                                 "interrupted": False, "replied": False, "ts": ts})
            self.pending_command = None

    def assistant(self, text, uses, mid):
        text = scrub(text or "")
        if self.rows and self.rows[-1]["role"] == "assistant" and mid and self.rows[-1].get("mid") == mid:
            row = self.rows[-1]
            row["text"] = "\n".join(t for t in (row["text"], text) if t)
            row["toolUses"] += uses
        else:
            self.rows.append({"role": "assistant", "text": text, "toolUses": uses, "mid": mid})
        for u in uses:
            self.uses[u["id"]] = u
        if self.prompts and (text.strip() or uses):
            self.prompts[-1]["replied"] = True

    def results(self, results):
        for b in results:
            use = self.uses.get(b.get("tool_use_id") or b.get("toolCallId"))
            if use and (b.get("is_error") or b.get("isError")):
                use["isError"] = True
                use["line"] = use["line"] if use["line"].endswith(" (error)") else use["line"] + " (error)"
        if self.rows and self.rows[-1]["role"] == "user" and self.rows[-1].get("toolResults"):
            return
        self.rows.append({"role": "user", "text": "", "toolUses": [], "toolResults": [{"n": len(results)}]})

    def record(self):
        if not self.prompts:
            return None
        rows = [{k: v for k, v in r.items() if k in ("role", "text", "toolUses", "toolResults")} for r in self.rows]
        for r in rows:
            r["toolUses"] = [{"tool": u["tool"], "line": u["line"], **({"isError": True} if u.get("isError") else {})}
                             for u in r["toolUses"]]
        key = hashlib.sha1(f"{self.source}/{self.path.name}".encode()).hexdigest()[:16]
        rec = {"key": key, "source": self.source, "file": self.path.name, "project": self.project,
               "session": self.session, "start": self.start, "end": self.end, "rows": rows,
               "prompts": self.prompts}
        hindsight(rec, self.rows)
        return rec


def hindsight(rec, raw_rows):
    """Per prompt: what the turn touched, and how it ties to what came before and after."""
    prompts = rec["prompts"]
    bounds = [p["row"] for p in prompts] + [len(raw_rows)]
    touched, seen_idents = set(), set()
    idents_by_prompt = []
    for i, p in enumerate(prompts):
        text = raw_rows[p["row"]]["text"]
        idents = set(IDENT.findall(text))
        idents_by_prompt.append(idents)
        read, edited = set(), set()
        for r in raw_rows[p["row"] + 1:bounds[i + 1]]:
            for u in r.get("toolUses") or []:
                name = (u["tool"] or "").lower()
                if u.get("path") and name in READ_TOOLS:
                    read.add(u["path"])
                if u.get("path") and name in EDIT_TOOLS:
                    edited.add(u["path"])
        files = read | edited
        p["hindsight"] = {
            "files_read": len(read), "files_edited": len(edited),
            "files_overlap_before": len(files & touched), "files_new": len(files - touched),
            "idents_reused": len(idents & seen_idents), "idents": len(idents),
        }
        touched |= files
        for r in raw_rows[bounds[i]:bounds[i + 1]]:
            seen_idents |= set(IDENT.findall(r["text"] or ""))
            for u in r.get("toolUses") or []:
                seen_idents |= set(IDENT.findall(u["line"]))
    for i, p in enumerate(prompts):
        before = set().union(*idents_by_prompt[:i]) if i else set()
        later = idents_by_prompt[i + 1:i + 6]
        p["hindsight"]["later_refer_back"] = sum(1 for ids in later if (ids & before) - idents_by_prompt[i])


def parse_claude_code(path, source, project):
    s = Session(source, path, project)
    seen = set()
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict) or r.get("isSidechain"):
            continue
        s.session = s.session or r.get("sessionId")
        if s.project is None and isinstance(r.get("cwd"), str):
            s.project = r["cwd"]
        uid = r.get("uuid")
        if uid:
            if uid in seen:
                continue
            seen.add(uid)
        msg = r.get("message") if isinstance(r.get("message"), dict) else {}
        if r.get("type") == "user":
            s.stamp(r.get("timestamp"))
            content = msg.get("content")
            results = tool_results(content)
            if results or r.get("toolUseResult") is not None:
                s.results(results)
                continue
            if r.get("isCompactSummary") or r.get("isVisibleInTranscriptOnly"):
                continue
            s.user_text(block_text(content), r.get("timestamp"), meta=bool(r.get("isMeta")))
        elif r.get("type") == "assistant":
            if msg.get("model") == "<synthetic>":
                continue
            s.stamp(r.get("timestamp"))
            uses = [tool_use(b.get("name"), b.get("input"), b.get("id")) for b in msg.get("content") or []
                    if isinstance(b, dict) and b.get("type") == "tool_use"]
            s.assistant(block_text(msg.get("content")), uses, msg.get("id") or uid)
    return s.record()


def parse_pi(path, source, project):
    s = Session(source, path, project)
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        if r.get("type") == "session":
            s.session = r.get("id")
            s.project = s.project or r.get("cwd")
        if r.get("type") != "message":
            continue
        msg = r.get("message") if isinstance(r.get("message"), dict) else {}
        role = msg.get("role")
        s.stamp(r.get("timestamp"))
        if role == "user":
            s.user_text(block_text(msg.get("content")), r.get("timestamp"))
        elif role == "assistant":
            uses = [tool_use(b.get("name"), b.get("arguments"), b.get("id")) for b in msg.get("content") or []
                    if isinstance(b, dict) and b.get("type") == "toolCall"]
            s.assistant(block_text(msg.get("content")), uses, r.get("id"))
        elif role == "toolResult":
            s.results([{"toolCallId": msg.get("toolCallId"), "isError": msg.get("isError")}])
    return s.record()


IDE_REQUEST = re.compile(r"^# Context from my IDE setup:.*?## My request for Codex:\s*", re.S)


def parse_codex(path, source, project):
    """Codex CLI rollouts: prompts from `event_msg/user_message`, the agent's side from
    `response_item` messages and tool calls. Reasoning, environment context and
    compaction records are left out."""
    s = Session(source, path, project)
    for line in path.open(errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        kind, p = r.get("type"), r.get("payload") if isinstance(r.get("payload"), dict) else {}
        if kind == "session_meta":
            s.session = s.session or p.get("id")
            s.project = s.project or p.get("cwd")
            continue
        s.stamp(r.get("timestamp"))
        if kind == "event_msg" and p.get("type") == "user_message":
            s.user_text(IDE_REQUEST.sub("", p.get("message") or ""), r.get("timestamp"))
        elif kind == "event_msg" and p.get("type") == "turn_aborted":
            if s.prompts:
                s.prompts[-1]["interrupted"] = True
        elif kind == "response_item":
            t = p.get("type")
            if t == "message" and p.get("role") == "assistant":
                text = "\n".join(b.get("text", "") for b in p.get("content") or []
                                 if isinstance(b, dict) and b.get("type") in ("output_text", "text"))
                s.assistant(text, [], None)
            elif t in ("function_call", "custom_tool_call", "local_shell_call"):
                args = p.get("arguments") if t == "function_call" else p.get("input") or p.get("action")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"input": args}
                uid = p.get("call_id") or p.get("id")
                s.assistant("", [tool_use(p.get("name") or t, args, uid)], s.rows[-1].get("mid")
                            if s.rows and s.rows[-1]["role"] == "assistant" else None)
            elif t in ("function_call_output", "custom_tool_call_output"):
                s.results([{"toolCallId": p.get("call_id")}])
    return s.record()


def detect(path):
    with path.open(errors="ignore") as f:
        head = f.read(20000)
    if re.search(r'"type"\s*:\s*"session"', head) and '"parentUuid"' not in head:
        return parse_pi
    if '"sessionId"' in head or '"parentUuid"' in head:
        return parse_claude_code
    if '"session_meta"' in head or '"response_item"' in head:
        return parse_codex
    return None


def files(sources):
    if sources in ("public", "both"):
        for path in sorted(RAW.rglob("*.jsonl")):
            yield path, path.relative_to(RAW).parts[0].replace("__", "/"), None
    if sources in ("local", "both"):
        # Read only: nothing under ~/.claude is opened for writing or copied out of data/.
        for path in sorted(LOCAL.rglob("*.jsonl")):
            if "subagents" in path.parts:
                continue
            yield path, "local", path.parent.name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", choices=["public", "local", "both"], default="public")
    parser.add_argument("--limit-sessions", type=int, help="Stop after this many sessions with prompts")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    load_env()

    counts = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for path, source, project in files(args.sources):
            parser_fn = detect(path)
            if not parser_fn:
                counts["skipped_files"] += 1
                continue
            try:
                rec = parser_fn(path, source, project)
            except Exception as e:  # One malformed export must not stop the run.
                print(f"{path}: {e}", file=sys.stderr)
                counts["failed_files"] += 1
                continue
            if not rec:
                counts["no_prompt_files"] += 1
                continue
            counts["sessions"] += 1
            counts["prompts"] += len(rec["prompts"])
            counts["rows"] += len(rec["rows"])
            counts["preceded_by_clear"] += sum(p["preceded_by"] == "clear" for p in rec["prompts"])
            counts["preceded_by_compact"] += sum(p["preceded_by"] == "compact" for p in rec["prompts"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.limit_sessions and counts["sessions"] >= args.limit_sessions:
                break
    print(json.dumps(dict(counts)), "->", args.out)


if __name__ == "__main__":
    main()
