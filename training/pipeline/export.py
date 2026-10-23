"""Write tev1 JSONL from the human labels of the review site, one file per session split.

    uv run python pipeline/export.py       # data/labels/human.jsonl -> data/export/{train,dev,test}.jsonl

A record is the tev1 example: the fixed system prompt, the JSON decision task the
student sees (pipeline/task.py) and the option letter a person picked. Skips and
"summary insufficient" are left out; the last answer for a point wins. The split is the
point's, which is by session.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, load_env  # noqa: E402
from task import CATEGORIES, LETTER_OF, messages  # noqa: E402


def records(human, points, states):
    latest = {}
    for r in human:
        latest[r["id"]] = r
    out = []
    for pid, r in latest.items():
        if r["label"] not in CATEGORIES or pid not in states or pid not in points:
            continue
        out.append({"id": pid, "split": points[pid]["split"], "session": points[pid]["session_key"],
                    "category": r["label"], "mode": r["mode"], "prompt": messages(states[pid]),
                    "completion": LETTER_OF[r["label"]]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", type=Path, default=DATA / "labels" / "human.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "export")
    args = parser.parse_args()
    load_env()

    read = lambda p: [json.loads(line) for line in p.open()] if p.exists() else []
    points = {p["id"]: p for p in read(DATA / "points.jsonl")}
    states = {s["id"]: s["state"] for s in read(DATA / "states.jsonl")}
    recs = records(read(args.human), points, states)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        with (args.out / f"{split}.jsonl").open("w") as f:
            for r in recs:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"exported {len(recs)} records {dict(Counter(r['split'] for r in recs))} -> {args.out}")
    for r in recs[:1]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
