"""Splices from held-out (test) sessions, for evaluation only.

    uv run python pipeline/heldout_splices.py            # -> data/points-test-splices.jsonl
    uv run python pipeline/build_tree.py --points data/points-test-splices.jsonl --out data/states-test-splices.jsonl
    uv run python pipeline/teacher.py --states data/states-test-splices.jsonl --splits test-splices \
        --out data/labels/mammoth-test-splices.jsonl
    uv run python pipeline/heldout_splices.py --join     # -> data/split/test-splices.jsonl

The test split holds almost no weakly labeled point (splices are train-only), so the
precision of `independent` against weak labels can't be read there. These points join
the end of one test session to the first prompt of another (same project when there is
one) and carry the weak label `new_task`. They are never trained on and never used to
pick a checkpoint: only pipeline/evaluate.py --split test-splices reads them.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import point_id, session_messages, weak_label  # noqa: E402
from config import DATA, load_env  # noqa: E402

LIMIT = 120


def make(points_path, out):
    points = [json.loads(line) for line in points_path.open()]
    test_keys = {p["session_key"] for p in points if p["split"] == "test"} | \
                {p["context_key"] for p in points if p["split"] == "test"}
    sessions = {}
    for line in (DATA / "clean.jsonl").open():
        s = json.loads(line)
        if s["key"] in test_keys:
            sessions[s["key"]] = s
    rng = random.Random(7)
    msgs = {k: session_messages(s) for k, s in sessions.items()}
    firsts = [s for s in sessions.values() if s["prompts"] and s["prompts"][0]["replied"]
              and msgs[s["key"]][1].get(s["prompts"][0]["row"]) == 0]
    ends = [s for s in sessions.values() if len(msgs[s["key"]][0]) >= 4]
    rng.shuffle(firsts)
    by_project = defaultdict(list)
    for x in ends:
        by_project[(x["source"], x.get("project"))].append(x)
    out_points, used = [], set()
    for y in firsts:
        pool = [x for x in by_project.get((y["source"], y.get("project")), []) if x["key"] not in (y["key"], *used)]
        pool = pool or [x for x in ends if x["source"] == y["source"] and x["key"] not in (y["key"], *used)]
        pool = pool or [x for x in ends if x["key"] not in (y["key"], *used)]
        if not pool:
            continue
        x = rng.choice(pool)
        used.add(x["key"])
        message = msgs[y["key"]][0][0]["text"]
        out_points.append({
            "id": point_id("heldout-splice", x["key"], y["key"]), "split": "test-splices", "session_key": y["key"],
            "context_key": x["key"], "t": len(msgs[x["key"]][0]), "message": message, "synthetic": "splice",
            "linked": None, "preceded_by": None, "source": y["source"],
            "same_project": x.get("project") == y.get("project"), "hindsight": None,
            "weak": weak_label(message, "splice", None)})
        if len(out_points) >= LIMIT:
            break
    with out.open("w") as f:
        for p in out_points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(out_points)} held-out splices ({sum(p['same_project'] for p in out_points)} same project) -> {out}")


def join():
    from dataset import example

    read = lambda p: [json.loads(line) for line in p.open()]
    points = {p["id"]: p for p in read(DATA / "points-test-splices.jsonl")}
    states = {s["id"]: s["state"] for s in read(DATA / "states-test-splices.jsonl")}
    labels = {lab["id"]: lab for lab in read(DATA / "labels" / "mammoth-test-splices.jsonl")}
    out = DATA / "split" / "test-splices.jsonl"
    with out.open("w") as f:
        for pid, p in points.items():
            if pid in states and pid in labels:
                f.write(json.dumps(example(p, states[pid], labels[pid]), ensure_ascii=False) + "\n")
    print(f"-> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--join", action="store_true")
    args = parser.parse_args()
    load_env()
    if args.join:
        join()
    else:
        make(DATA / "points.jsonl", DATA / "points-test-splices.jsonl")


if __name__ == "__main__":
    main()
