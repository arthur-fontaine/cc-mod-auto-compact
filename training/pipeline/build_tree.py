"""Build the summary tree nodes every decision point needs, then the points' states.

    uv run python pipeline/build_tree.py --dry-run             # nodes, calls and spend estimate
    uv run python pipeline/build_tree.py                       # data/points.jsonl -> data/states.jsonl

Only the nodes the points' views need are built: their lines, the lines those merge,
and each summary call's own <chat> context. A node that fits in 512 bytes costs no
call. Every built node is cached by hash(prompt version, model, effort, request) in
data/tree-cache/, so a rerun with the same inputs makes no Haiku call.

A Claude usage limit stops every Haiku call: the cache is kept, a notification is
shown, a CLAUDE USAGE STOP block gives the reset time and the resume command, and the
states of points whose nodes are all built are still written.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarizer  # noqa: E402
import tree  # noqa: E402
from config import DATA, guard_spend, load_env  # noqa: E402
from task import agent_reply, clip_state, clip_state_v2, state_size  # noqa: E402
from text import estimate_tokens, render, utf8_len  # noqa: E402

CACHE = DATA / "tree-cache" / "nodes.jsonl"
RESUME = "cd training && uv run python pipeline/build_tree.py"
SYSTEM_TOKENS = 900


class NodeCache:
    def __init__(self, path=CACHE):
        self.path = path
        self.entries = {}
        if path.exists():
            for line in path.open():
                e = json.loads(line)
                self.entries[e["key"]] = e

    def put(self, entry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.entries[entry["key"]] = entry


def node_key(prompt):
    s = summarizer.settings()
    raw = "\n".join([summarizer.prompt_version(), s["model"], s["effort"], prompt])
    return hashlib.sha256(raw.encode()).hexdigest()


async def summarize(prompt, complete=None, limit=tree.LIMIT):
    """One node through Haiku with up to tree.RETRIES "too long" retries: (line, calls, list_usd)."""
    complete = complete or summarizer.complete
    model = summarizer.settings()["model"]
    lines, calls, cost = [], 0, 0.0
    ask = prompt
    for _ in range(1 + tree.RETRIES):
        text, usage = await complete(ask)
        calls += 1
        cost += summarizer.list_price(usage, model)
        line = tree.clean_line(text)
        if line:
            lines.append(line)
            if utf8_len(line) <= limit:
                break
            ask = tree.retry_prompt(prompt, line, limit)
    if not lines:
        raise RuntimeError("the summarizer returned no text")
    return tree.shortest(lines), calls, cost


class V1:
    """The v1 state: the line-budget view and the 512-byte lines."""
    limit = tree.LIMIT
    context = staticmethod(tree.context_view)
    needed = staticmethod(tree.needed)

    def lines(self, msgs, t):
        return tree.view(t), None

    def state(self, conversation, last, message):
        return clip_state(conversation, message)


class V2:
    """The v2 state (PLAN.md section 2): batched memory, the agent's last reply raw."""
    context = staticmethod(tree.memory_context)
    needed = staticmethod(tree.needed_memory)

    def __init__(self, limit):
        self.limit = limit

    def lines(self, msgs, t):
        last = msgs[t - 1] if t and msgs[t - 1]["kind"] == "agent" else None
        return tree.memory_view(t - 1 if last else t), last

    def state(self, memory, last, message):
        return clip_state_v2(memory, agent_reply(last), message)


class Builder:
    def __init__(self, cache, concurrency, complete=None, fmt=None):
        self.fmt = fmt or V1()
        self.cache = cache
        self.sem = asyncio.Semaphore(concurrency)
        self.complete = complete
        self.tasks = {}
        self.calls = self.new_nodes = 0
        self.list_usd = 0.0
        self.stopped = None

    def node(self, session, msgs, node):
        key = (session, node)
        if key not in self.tasks:
            self.tasks[key] = asyncio.ensure_future(self._build(session, msgs, node))
        return self.tasks[key]

    async def text(self, session, msgs, node):
        return await self.node(session, msgs, node)

    async def _build(self, session, msgs, node):
        context = self.fmt.context(node)
        if node[0] == 0:
            source = render(msgs[node[1]])
        else:
            a, b = tree.children(node)
            source = (await self.text(session, msgs, a), await self.text(session, msgs, b))
        free = tree.no_call_text(node, source, self.fmt.limit)
        if free is not None:
            return free
        ctx = [(n, await self.text(session, msgs, n)) for n in context]
        if node[0] == 0:
            prompt = tree.leaf_prompt(node[1], source, ctx)
        else:
            prompt = tree.merge_prompt(a, b, source[0], source[1], ctx)
        key = node_key(prompt)
        hit = self.cache.entries.get(key)
        if hit:
            return hit["text"]
        if self.stopped:
            raise self.stopped
        async with self.sem:
            if self.stopped:
                raise self.stopped
            try:
                line, calls, cost = await summarize(prompt, self.complete, self.fmt.limit)
            except summarizer.UsageLimit as e:
                self.stopped = e
                raise
        self.calls += calls
        self.list_usd += cost
        self.new_nodes += 1
        self.cache.put({"key": key, "text": line, "calls": calls, "list_usd": round(cost, 6)})
        return line


def plan(points, sessions, fmt=None):
    """{context session: (messages, {t: (view, last agent message or None)})} for every point."""
    from text import messages
    from build import MAX_MESSAGES

    need = defaultdict(set)
    for p in points:
        need[p["context_key"]].add(p["t"])
    out = {}
    for key, ts in need.items():
        msgs = messages(sessions[key]["rows"])[:MAX_MESSAGES]
        out[key] = (msgs, {t: (fmt or V1()).lines(msgs, t) for t in sorted(ts)})
    return out


def estimate(planned, cache, fmt=None):
    """(nodes, upper estimate of calls, list USD) for the nodes not yet in the cache."""
    fmt = fmt or V1()
    nodes = calls = 0
    tokens_in = 0
    for key, (msgs, views) in planned.items():
        for node in fmt.needed([v for v, _ in views.values()]):
            nodes += 1
            if node[0] == 0:
                size = utf8_len(render(msgs[node[1]]))
                if size <= fmt.limit:
                    continue
            else:
                size = 2 * fmt.limit
            calls += 1
            tokens_in += SYSTEM_TOKENS + estimate_tokens("x" * (size + 16 * fmt.limit))
    calls = int(calls * 1.1)  # "too long" retries
    inp, out = summarizer.PRICE[summarizer.settings()["model"]]
    return nodes, calls, (tokens_in * 1.1 * inp + calls * 200 * out) / 1e6


async def run(planned, points, builder):
    async def one(p):
        msgs, views = planned[p["context_key"]]
        lines, last = views[p["t"]]
        try:
            texts = [await builder.text(p["context_key"], msgs, n) for n in lines]
        except summarizer.UsageLimit:
            return None
        except Exception as e:
            print(f"point {p['id']}: {e}", file=sys.stderr, flush=True)
            return None
        conversation = tree.view_text(list(zip(lines, texts)))
        return builder.fmt.state(conversation, last, p["message"])

    started, done = time.time(), 0
    results = {}

    async def tracked(p):
        nonlocal done
        results[p["id"]] = await one(p)
        done += 1
        if done % 100 == 0:
            print(f"[tree] {done}/{len(points)} points · {builder.calls} calls · {builder.new_nodes} new nodes · "
                  f"{time.time() - started:.0f}s", flush=True)

    await asyncio.gather(*(tracked(p) for p in points))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, default=DATA / "points.jsonl")
    parser.add_argument("--sessions", type=Path, default=DATA / "clean.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "states.jsonl")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=["v1", "v2"], default="v1")
    parser.add_argument("--limit", type=int, default=tree.MEMORY_LIMIT, help="v2: the line limit in bytes")
    parser.add_argument("--split", help="Only the points of this split")
    args = parser.parse_args()
    load_env()
    fmt = V1()
    if args.format == "v2":
        fmt = V2(args.limit)
        summarizer.system_override = summarizer.system_prompt_v2(args.limit)
        if args.out == DATA / "states.jsonl":
            args.out = DATA / f"states-v2-{args.limit}.jsonl"

    print(f"[tree] auth mode: {summarizer.auth_mode()}", flush=True)
    points = [json.loads(line) for line in args.points.open()]
    if args.split:
        points = [p for p in points if p["split"] == args.split]
    if args.limit_sessions:
        keep = set(list(dict.fromkeys(p["session_key"] for p in points))[: args.limit_sessions])
        points = [p for p in points if p["session_key"] in keep]
    sessions = {}
    wanted = {p["context_key"] for p in points}
    for line in args.sessions.open():
        s = json.loads(line)
        if s["key"] in wanted:
            sessions[s["key"]] = s
    planned = plan(points, sessions, fmt)
    cache = NodeCache()
    nodes, calls, list_usd = estimate(planned, cache, fmt)
    billed = list_usd if summarizer.auth_mode() == "API" else 0.0
    print(f"[tree] {len(points)} points over {len(planned)} sessions need {nodes} nodes", flush=True)
    if not guard_spend("haiku", calls=calls, billed_usd=billed, list_usd=list_usd, dry_run=args.dry_run):
        return

    concurrency = int(os.environ.get("SUMMARIZER_CONCURRENCY") or 8)
    builder = Builder(cache, concurrency, fmt=fmt)
    results = asyncio.run(run(planned, points, builder))
    written = 0
    with args.out.open("w") as f:
        for p in points:
            state = results.get(p["id"])
            if state is None:
                continue
            written += 1
            f.write(json.dumps({"id": p["id"], "split": p["split"], "session_key": p["session_key"],
                                "state": state, **state_size(state)}, ensure_ascii=False) + "\n")
    billed_done = builder.list_usd if summarizer.auth_mode() == "API" else 0.0
    print(f"[tree] spend: {builder.calls} Haiku calls, {builder.new_nodes} new nodes, list-price equivalent "
          f"${builder.list_usd:.2f}, billed ${billed_done:.2f}; {written}/{len(points)} states -> {args.out}",
          flush=True)
    if builder.stopped:
        summarizer.notify_usage_stop()
        print(summarizer.usage_stop_block(builder.stopped, RESUME))
        sys.exit(5)


if __name__ == "__main__":
    main()
