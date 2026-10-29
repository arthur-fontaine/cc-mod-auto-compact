"""Where a decision's tokens come from, with the student's own tokenizer.

    uv run python pipeline/tokens.py --gguf PATH [--split test --split test-splices]   # -> results/tokens-SPLIT.json
    uv run python pipeline/tokens.py --report NAME [--report-splices NAME-splices]       # accuracy by token bin
    uv run python pipeline/tokens.py --gguf PATH --states data/states-v2-160.jsonl --format v2   # v2 states

Each state is rendered as the student reads it (task.messages, in Qwen's chat markup) three
times: whole, without the new message, and without the conversation either. The differences
give the conversation's and the message's tokens. `reused` is the prefix shared with the
previous state of the same session (what llama-server's one-slot cache keeps without help);
`warm_left` is what is left to compute once the state without the message was prefilled.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import messages, messages_v2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BINS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, None)]


def render(state):
    system, user = (messages_v2 if "memory" in state else messages)(state)
    return (f"<|im_start|>system\n{system['content']}<|im_end|>\n<|im_start|>user\n{user['content']}<|im_end|>\n"
            "<|im_start|>assistant\n")


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def line_kind(line):
    """`merged` for a line over several messages, else the first word of a one-message line."""
    head, _, body = line.partition("|")
    if "+" in head and head.split("+")[1].isdigit() and int(head.split("+")[1]) > 1:
        return "merged"
    m = re.match(r"(\w+):", body)
    return m.group(1) if m else "other"


def bin_of(total):
    for lo, hi in BINS:
        if total >= lo and (hi is None or total < hi):
            return f"{lo}-{hi}" if hi else f"{lo}+"


def breakdown(state, tokenize, previous=None):
    """Token counts of one state. For a v2 state, `conversation` counts the memory and
    `agent` the agent's last reply."""
    v2 = "memory" in state
    memory = state["memory"] if v2 else state["conversation"]
    full = tokenize(render(state))
    no_msg = tokenize(render({**state, "latest_user_message": ""}))
    empty = {"memory": "", "last_agent_message": "", "latest_user_message": ""} if v2 else \
        {"conversation": "", "latest_user_message": ""}
    fixed = len(tokenize(render(empty)))
    agent = len(no_msg) - len(tokenize(render({**state, "last_agent_message": "", "latest_user_message": ""}))) \
        if v2 else 0
    lines = memory.split("\n") if memory else []
    kinds = {}
    for line in lines:
        k = kinds.setdefault(line_kind(line), [0, 0])
        k[0] += 1
        k[1] += len(line.encode())
    return {"total": len(full), "fixed": fixed, "conversation": len(no_msg) - fixed - agent, "agent": agent,
            "message": len(full) - len(no_msg), "lines": len(lines),
            "kinds": kinds, "reused": common_prefix(previous, full) if previous else None,
            "warm_left": len(full) - common_prefix(no_msg, full)}, full


def llama_tokenizer(gguf):
    tmp = Path(tempfile.mkdtemp())

    def tokenize(text):
        path = tmp / "text.txt"
        path.write_text(text)
        out = subprocess.run(["llama-tokenize", "-m", str(gguf), "-f", str(path), "--ids", "--log-disable",
                              "--no-bos"], capture_output=True, text=True, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])
    return tokenize


def measure(path, tokenize):
    rows, last = [], {}
    for line in Path(path).open():
        e = json.loads(line)
        session = e.get("session") or e.get("session_key")
        out, ids = breakdown(e["state"], tokenize, last.get(session) if session else None)
        if session:
            last[session] = ids
        rows.append({"id": e["id"], "session": session, **out})
    return rows


def report(pred_names):
    import benchmark
    tokens = {}
    for path in ROOT.glob("results/tokens-*.json"):
        tokens.update({r["id"]: r for r in json.loads(path.read_text())})
    g = benchmark.gold()
    s = benchmark.score(benchmark.predictions(*pred_names), g)
    by = {}
    for r in s["rows"]:
        if r["id"] in tokens:
            by.setdefault(bin_of(tokens[r["id"]]["total"]), []).append(r)
    out = {}
    for lo, hi in BINS:
        b = f"{lo}-{hi}" if hi else f"{lo}+"
        rows = by.get(b, [])
        act = [r for r in rows if r["gold_action"] != "keep"]
        cats = [r for r in rows if r["gold_category"]]
        ms = sorted(r["ms"] for r in rows if r["ms"] is not None)
        out[b] = {"n": len(rows),
                  "right_action": sum(r["action"] == r["gold_action"] for r in rows) / len(rows) if rows else None,
                  "acted": [sum(r["action"] == r["gold_action"] for r in act), len(act)],
                  "right_category": sum(r["category"] == r["gold_category"] for r in cats) / len(cats) if cats else None,
                  "p50_ms": ms[len(ms) // 2] if ms else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", help="The student's GGUF, for its tokenizer")
    parser.add_argument("--split", action="append")
    parser.add_argument("--states", type=Path, help="A states file (build_tree.py output) instead of a split")
    parser.add_argument("--report", help="Predictions name on test, scored by token bin")
    parser.add_argument("--report-splices", help="Predictions name on the held-out splices")
    args = parser.parse_args()
    if args.report:
        print(json.dumps(report((args.report, args.report_splices)), indent=1))
        return
    if not args.gguf:
        parser.error("give --gguf or --report")
    tokenize = llama_tokenizer(args.gguf)
    if args.states:
        rows = measure(args.states, tokenize)
        (ROOT / "results" / f"tokens-{args.states.stem}.json").write_text(json.dumps(rows))
        print(args.states.stem, len(rows), "states", file=sys.stderr)
        return
    for split in args.split or ["test", "test-splices"]:
        rows = measure(ROOT / "data" / "split" / f"{split}.jsonl", tokenize)
        (ROOT / "results" / f"tokens-{split}.json").write_text(json.dumps(rows))
        print(split, len(rows), "states", file=sys.stderr)


if __name__ == "__main__":
    main()
