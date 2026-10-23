"""Pick decision points, split them by session, and add synthetic splices.

    uv run python pipeline/build.py      # data/clean.jsonl -> data/points.jsonl + data/build-report.json

A decision point is a human prompt with at least MIN_CONTEXT messages before it in its
session (capped at MAX_MESSAGES, which bounds the tree's cost). Its state will be the
view of those messages and the prompt.

- `/clear` starts a new session file, so the context the user cleared is the previous
  session of the same project. When one ended shortly before, the first prompt after
  the clear is a real point on that context (`linked: clear`), and both sessions are
  split together.
- Splits are by session (linked sessions as one group): test-candidates, dev, train.
  Each session gives at most 40 train points or 12 held-out points.
- Splices: the end of session X followed by the first prompt of session Y from the same
  project (or the same dataset), marked `synthetic: splice`, train only.
- Weak labels for the report: prompts after /clear and splices should be
  independent/related; short acknowledgements should be continuation.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, load_env  # noqa: E402
from text import messages  # noqa: E402

MAX_MESSAGES = 200
MIN_CONTEXT = 2
CAPS = {"train": 40, "dev": 12, "test": 12}
TEST_FRACTION, DEV_FRACTION = 0.12, 0.08
SPLICE_RATE = 0.15  # splices per real train point
LINK_GAP_S = 6 * 3600
ACK = re.compile(r"^\s*(y(es|ep|eah)?|ok(ay)?|sure|thanks?( you)?|thx|go( ahead)?|do it|continue|proceed|lgtm|"
                 r"sounds good|perfect|great|nice|cool|please do|yes please|go on|keep going)[\s.!]*$", re.I)


def bucket(key):
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10_000 / 10_000


def split_of(group_key):
    b = bucket(group_key)
    return "test" if b < TEST_FRACTION else "dev" if b < TEST_FRACTION + DEV_FRACTION else "train"


def ts(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def weak_label(message, synthetic, preceded_by):
    if synthetic or preceded_by == "clear":
        return "new_task"  # expected: independent or related
    if ACK.match(message) and len(message) <= 40:
        return "continuation"
    return None


def point_id(*parts):
    return hashlib.sha1("/".join(map(str, parts)).encode()).hexdigest()[:16]


def session_messages(s):
    msgs = messages(s["rows"])[:MAX_MESSAGES]
    index = {m["row"]: i for i, m in enumerate(msgs) if m["kind"] == "user"}
    return msgs, index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DATA / "clean.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "points.jsonl")
    parser.add_argument("--limit-sessions", type=int)
    args = parser.parse_args()
    load_env()

    sessions = [json.loads(line) for line in args.input.open()][: args.limit_sessions]
    by_key = {s["key"]: s for s in sessions}
    msgs = {s["key"]: session_messages(s) for s in sessions}
    report = Counter()

    # Link each session that starts after /clear to the previous session of its project.
    parent = {k: k for k in by_key}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    by_project = defaultdict(list)
    for s in sessions:
        if s.get("project") and ts(s.get("start") or "") and ts(s.get("end") or ""):
            by_project[(s["source"], s["project"])].append(s)
    links = {}
    for group in by_project.values():
        group.sort(key=lambda s: ts(s["start"]))
        for s in group:
            first = s["prompts"][0]
            if first["preceded_by"] != "clear" or msgs[s["key"]][1].get(first["row"]) != 0:
                continue
            start = ts(s["start"])
            before = [x for x in group if x["key"] != s["key"] and ts(x["end"]) <= start
                      and start - ts(x["end"]) <= LINK_GAP_S and len(msgs[x["key"]][0]) >= MIN_CONTEXT]
            if before:
                x = max(before, key=lambda x: ts(x["end"]))
                links[s["key"]] = x["key"]
                parent[root(s["key"])] = root(x["key"])
    report["clear_links"] = len(links)

    rng = random.Random(42)
    points = []
    for s in sessions:
        split = split_of(f"{by_key[root(s['key'])]['source']}/{root(s['key'])}")
        ms, index = msgs[s["key"]]
        candidates = []
        for p in s["prompts"]:
            t = index.get(p["row"])
            if t is None:
                report["beyond_message_cap"] += 1
                continue
            if t == 0 and s["key"] in links and p.get("point", True) is not False:
                ctx = links[s["key"]]
                candidates.append({"context_key": ctx, "t": len(msgs[ctx][0]), "linked": "clear", "p": p,
                                   "message": ms[t]["text"]})
                continue
            if not p.get("point") or t < MIN_CONTEXT:
                report["not_eligible"] += 1
                continue
            candidates.append({"context_key": s["key"], "t": t, "linked": None, "p": p, "message": ms[t]["text"]})
        rng.shuffle(candidates)
        if len(candidates) > CAPS[split]:
            report[f"{split}_session_cap"] += len(candidates) - CAPS[split]
        for c in candidates[: CAPS[split]]:
            p = c["p"]
            points.append({
                "id": point_id(s["key"], c["context_key"], c["t"]), "split": split, "session_key": s["key"],
                "context_key": c["context_key"], "t": c["t"], "message": c["message"], "synthetic": None,
                "linked": c["linked"], "preceded_by": p["preceded_by"], "source": s["source"],
                "hindsight": p.get("hindsight"), "weak": weak_label(c["message"], None, p["preceded_by"]),
            })

    # Splices: train sessions only, never in held-out splits.
    train = [s for s in sessions if split_of(f"{by_key[root(s['key'])]['source']}/{root(s['key'])}") == "train"]
    real_train = sum(p["split"] == "train" for p in points)
    want = int(real_train * SPLICE_RATE)
    firsts = [s for s in train if s["prompts"] and s["prompts"][0].get("point", True)
              and msgs[s["key"]][1].get(s["prompts"][0]["row"]) == 0 and s["prompts"][0]["replied"]
              and s["key"] not in links]
    ends = [s for s in train if len(msgs[s["key"]][0]) >= 4]
    rng.shuffle(firsts)
    rng.shuffle(ends)
    by_source_project = defaultdict(list)
    for x in ends:
        by_source_project[(x["source"], x.get("project"))].append(x)
    used_x = set()
    for y in firsts:
        if sum(p["synthetic"] == "splice" for p in points) >= want:
            break
        pool = [x for x in by_source_project.get((y["source"], y.get("project")), []) if x["key"] != y["key"]]
        pool = [x for x in pool if x["key"] not in used_x] or \
               [x for x in ends if x["source"] == y["source"] and x["key"] not in used_x and x["key"] != y["key"]]
        if not pool:
            continue
        x = pool[0]
        used_x.add(x["key"])
        message = msgs[y["key"]][0][0]["text"]
        points.append({
            "id": point_id("splice", x["key"], y["key"]), "split": "train", "session_key": y["key"],
            "context_key": x["key"], "t": len(msgs[x["key"]][0]), "message": message, "synthetic": "splice",
            "linked": None, "preceded_by": y["prompts"][0]["preceded_by"], "source": y["source"],
            "same_project": x.get("project") == y.get("project"), "hindsight": None,
            "weak": weak_label(message, "splice", None),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary = {
        "sessions": len(sessions),
        "points": dict(Counter(p["split"] for p in points)),
        "splices": sum(p["synthetic"] == "splice" for p in points),
        "linked_clear": sum(p["linked"] == "clear" for p in points),
        "weak": dict(Counter(f"{p['split']}:{p['weak']}" for p in points if p["weak"])),
        "sessions_per_split": {k: len({p["session_key"] for p in points if p["split"] == k})
                               for k in ("train", "dev", "test")},
        "filtered": dict(report),
    }
    (args.out.parent / "build-report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
