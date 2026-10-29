"""A tuning sample of train, to choose a state format without wearing out dev (PLAN.md 3).

    uv run python pipeline/tune_split.py     # -> data/split/tune.jsonl, data/points-tune.jsonl

Whole train sessions drawn at random (fixed seed) until TARGET real points, plus the train
splices whose context is one of them, so the sample has points that act. Train sessions are
the student's own, never held out, so tuning on them leaves dev and test as they were.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA  # noqa: E402

TARGET = 800
SEED = 0


def sample(points, target=TARGET, seed=SEED):
    """The ids of the sample: real points of drawn sessions, then splices onto them."""
    real = [p for p in points if p["context_key"] == p["session_key"]]
    sessions = sorted({p["session_key"] for p in real})
    random.Random(seed).shuffle(sessions)
    chosen, n = set(), 0
    count = {}
    for p in real:
        count[p["session_key"]] = count.get(p["session_key"], 0) + 1
    for s in sessions:
        if n >= target:
            break
        chosen.add(s)
        n += count[s]
    return {p["id"] for p in points if p["context_key"] in chosen}


def main():
    points = [p for p in map(json.loads, (DATA / "points.jsonl").open()) if p["split"] == "train"]
    ids = sample(points)
    with (DATA / "points-tune.jsonl").open("w") as f:
        for p in points:
            if p["id"] in ids:
                f.write(json.dumps({**p, "split": "tune"}, ensure_ascii=False) + "\n")
    rows = [r for r in map(json.loads, (DATA / "split" / "train.jsonl").open()) if r["id"] in ids]
    with (DATA / "split" / "tune.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({**r, "split": "tune"}, ensure_ascii=False) + "\n")
    splices = sum(1 for p in points if p["id"] in ids and p["context_key"] != p["session_key"])
    print(f"{len(ids)} points ({len(ids) - splices} real, {splices} splices) over "
          f"{len({p['context_key'] for p in points if p['id'] in ids})} sessions; {len(rows)} in train.jsonl")


if __name__ == "__main__":
    main()
