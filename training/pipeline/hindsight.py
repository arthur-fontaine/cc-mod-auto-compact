"""An answer key for the benchmark that Mammoth did not write: Claude teachers with hindsight.

    uv run python pipeline/hindsight.py --dry-run
    uv run python pipeline/hindsight.py --teacher claude-sonnet-5-5
    uv run python pipeline/hindsight.py --teacher claude-opus-5-5
    uv run python pipeline/hindsight.py --gold          # -> data/labels/hindsight-gold.jsonl
    uv run python pipeline/hindsight.py --split dev ...  # the same on dev, to choose a format (PLAN.md 3)

Mammoth is the teacher the student copies, so scoring Mammoth against itself says nothing.
As in Nisev's test set, two Claude teachers label each held-out test prompt independently,
seeing what the classifier sees (the summary lines and the new message) plus what it can't:
the raw recent messages, what the agent did next, and the user's next prompts. The gold
keeps the points where both agree.

Runs through `claude -p` on the Claude Code login of whoever runs it, isolated like the
summarizer (safe mode, no setting sources, tools, MCP servers or slash commands, no session
persistence), API keys stripped unless SUMMARIZER_USE_API=1. Texts were scrubbed by extract
and clean. Results append, so an interrupted run resumes.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, guard_spend, load_env  # noqa: E402
from policy import action  # noqa: E402
from task import CATEGORIES, CRITERIA  # noqa: E402
from text import clip, messages, render  # noqa: E402

BATCH = 10
OUT = DATA / "labels" / "hindsight-{teacher}.jsonl"
GOLD = DATA / "labels" / "hindsight-gold.jsonl"

SYSTEM = """You label the answer key of a benchmark for a classifier used with AI coding agents (Claude Code and similar).

For each new message a developer sends during a session, the classifier decides how much of the conversation the agent still needs, by putting the message in one category:

""" + "\n".join(f"- {c}: {d}" for c, d in CRITERIA.items()) + """

The agent then keeps the context (continuation, follow_up), compacts it to a summary (related, return) or clears it (independent). Keeping context that is no longer needed only costs tokens; dropping context that was needed makes the agent lose track. So `independent` means none of the conversation would have helped, and `related` that only the general picture would have.

For each item you get what the classifier sees (`summary`: one-line summaries of the conversation, oldest first; `message`: the new message) and hindsight it cannot see: `recent`, the last messages before the new one in full; `next_turn`, what the agent did after it; `next_prompts`, the developer's following messages; and `stats` (files the turn read and edited, how many of them the session had touched before, identifiers reused from earlier, later prompts that refer back to before this one). Use the hindsight to judge what the message really needed: whether the agent's work on it drew on the earlier conversation, and whether the developer came back to the earlier work.

`confidence` is your probability that the category is right (0 to 1). Keep `reason` under 20 words."""

# A key that no state format wrote: the teachers see no summary line, only the raw messages, so the
# key cannot favour the format whose lines they read (PLAN.md 3).
SYSTEM_NEUTRAL = SYSTEM.replace(
    "For each item you get what the classifier sees (`summary`: one-line summaries of the conversation, oldest "
    "first; `message`: the new message) and hindsight it cannot see: `recent`, the last messages before the new "
    "one in full;",
    "For each item you get `message`, the new message, and `recent`, the conversation's last messages before it "
    "in full, oldest first (`earlier` says how many older messages there were), plus hindsight:")

SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "category": {"type": "string", "enum": CATEGORIES},
                       "confidence": {"type": "number"}, "reason": {"type": "string"}},
        "required": ["id", "category", "confidence", "reason"]}}},
    "required": ["labels"],
}


def paths(split, neutral=False):
    """The labels and gold files of a split: the test key keeps its names; dev's say dev."""
    tag = ("" if split == "test" else f"-{split}") + ("-neutral" if neutral else "")
    return Path(str(OUT).replace("hindsight-", f"hindsight{tag}-")), GOLD.with_name(f"hindsight{tag}-gold.jsonl")


def items(splices=False, split="test", neutral=False):
    """Held-out test prompts, or with `splices` the held-out splices (pipeline/heldout_splices.py),
    or the prompts of another split."""
    suffix = "-test-splices" if splices else ""
    points = [json.loads(line) for line in (DATA / f"points{suffix}.jsonl").open()]
    points = [p for p in points if p["split"] in ((split,) if split != "test" else ("test", "test-splices"))]
    states = {s["id"]: s["state"] for s in map(json.loads, (DATA / f"states{suffix}.jsonl").open())}
    keys = {p["session_key"] for p in points} | {p["context_key"] for p in points}
    sessions = {}
    for line in (DATA / "clean.jsonl").open():
        s = json.loads(line)
        if s["key"] in keys:
            sessions[s["key"]] = s
    out = []
    for p in points:
        if p["id"] not in states:
            continue
        t = p["t"]
        before = messages(sessions[p["context_key"]]["rows"])[:t]
        own = messages(sessions[p["session_key"]]["rows"])
        # A splice's prompt opens the other session: what followed it is there.
        after = own[t + 1:] if p["context_key"] == p["session_key"] else own[1:]
        if neutral:
            out.append({"id": p["id"], "earlier": max(0, len(before) - 12),
                        "recent": [clip(render(m), 1200) for m in before[-12:]]})
        else:
            out.append({"id": p["id"], "summary": clip(states[p["id"]]["conversation"], 8000),
                        "recent": [clip(render(m), 1200) for m in before[-8:]]})
        out[-1].update({
            "message": clip(p["message"], 4000),
            "next_turn": clip(render(after[0]), 2500) if after and after[0]["kind"] == "agent" else None,
            "next_prompts": [clip(m["text"], 400) for m in after if m["kind"] == "user"][:3],
            "stats": p.get("hindsight"),
        })
    return out


def env():
    e = dict(os.environ)
    if os.environ.get("SUMMARIZER_USE_API") != "1":
        e.pop("ANTHROPIC_API_KEY", None)
        e.pop("ANTHROPIC_AUTH_TOKEN", None)
    return e


def call(teacher, batch, system=SYSTEM):
    prompt = f"Label these {len(batch)} items. Return one label per id.\n\n" + json.dumps(batch, ensure_ascii=False, indent=1)
    with tempfile.TemporaryDirectory() as cwd:
        done = subprocess.run(
            ["claude", "-p", "--model", teacher, "--safe-mode", "--setting-sources", "", "--strict-mcp-config",
             "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--no-session-persistence",
             "--tools", "", "--output-format", "json", "--system-prompt", system, "--json-schema", json.dumps(SCHEMA)],
            input=prompt, capture_output=True, text=True, cwd=cwd, timeout=900, env=env())
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout)[-400:])
    out = json.loads(done.stdout)
    result = out.get("structured_output") or json.loads(out["result"])
    return result["labels"], out.get("total_cost_usd") or 0.0


def label(teacher, dry_run, splices=False, split="test", neutral=False):
    path = Path(str(paths(split, neutral)[0]).format(teacher=teacher))
    done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
    todo = [i for i in items(splices, split, neutral) if i["id"] not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"[{teacher}] {len(done)} labeled, {len(todo)} to go in {len(batches)} calls (subscription)", flush=True)
    if not guard_spend(teacher, calls=len(batches), billed_usd=0.0, list_usd=None, dry_run=dry_run):
        return
    lock, cost, n = threading.Lock(), [0.0], [0]

    def run(batch):
        ids = {b["id"] for b in batch}
        for _ in range(3):
            try:
                labels, usd = call(teacher, batch, SYSTEM_NEUTRAL if neutral else SYSTEM)
                good = [lab for lab in labels if lab["id"] in ids and lab["category"] in CATEGORIES]
                with lock, path.open("a") as f:
                    for lab in good:
                        f.write(json.dumps({**lab, "teacher": teacher}, ensure_ascii=False) + "\n")
                    cost[0] += usd
                    n[0] += len(good)
                return
            except Exception as e:
                err = e
        print(f"batch failed: {str(err)[:300]}", file=sys.stderr, flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(run, batches))
    print(f"[{teacher}] spend: {len(batches)} calls, {n[0]} labels, list-price equivalent ${cost[0]:.2f}, "
          f"billed $0.00 (subscription)", flush=True)


def gold(teachers, split="test", neutral=False):
    out, gold_path = paths(split, neutral)
    read = lambda t: {lab["id"]: lab for lab in map(json.loads, Path(str(out).format(teacher=t)).open())}
    a, b = (read(t) for t in teachers)
    both = sorted(set(a) & set(b))
    rows = []
    for i in both:
        ca, cb = a[i]["category"], b[i]["category"]
        rows.append({"id": i, "category": ca if ca == cb else None, "action": action(ca) if action(ca) == action(cb) else None,
                     "teachers": {teachers[0]: ca, teachers[1]: cb}})
    gold_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cat = [r for r in rows if r["category"]]
    act = [r for r in rows if r["action"]]
    print(f"{len(both)} labeled by both · category agreement {len(cat) / len(both):.3f} ({len(cat)}) · "
          f"action agreement {len(act) / len(both):.3f} ({len(act)}) -> {gold_path}")
    print("gold categories:", dict(Counter(r["category"] for r in cat)))
    print("gold actions:", dict(Counter(r["action"] for r in act)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="claude-sonnet-5-5")
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--splices", action="store_true", help="Label the held-out splices instead")
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--neutral", action="store_true", help="Teachers see raw messages, no summary line")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_env()
    if args.gold:
        gold(["claude-sonnet-5-5", "claude-opus-5-5"], args.split, args.neutral)
    else:
        label(args.teacher, args.dry_run, args.splices, args.split, args.neutral)


if __name__ == "__main__":
    main()
