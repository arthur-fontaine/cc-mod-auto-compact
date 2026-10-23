"""Join states and Mammoth's labels into the student's train/dev/test files.

    uv run python pipeline/dataset.py      # -> data/split/{train,dev,test}.jsonl

Train is every Mammoth-labeled train point, unreviewed, synthetic splices included. dev
(checkpoint selection, calibration) and test (section 11) are real points from held-out
sessions only. Each example carries the tev1 prompt, the letter Mammoth picked and its
whole distribution, the soft target.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, load_env  # noqa: E402
from task import CATEGORIES, LETTER_OF, clip_state, messages  # noqa: E402


def example(point, state, label):
    # Through the shared clip again, so a state built by an older pipeline still fits.
    state = clip_state(state["conversation"], state["latest_user_message"])
    return {"id": point["id"], "split": point["split"], "session": point["session_key"],
            "synthetic": point["synthetic"], "weak": point.get("weak"), "state": state,
            "prompt": messages(state), "category": label["choice"], "completion": LETTER_OF[label["choice"]],
            "targets": [label["probabilities"][c] for c in CATEGORIES]}


def main():
    load_env()
    read = lambda p: [json.loads(line) for line in p.open()]
    points = {p["id"]: p for p in read(DATA / "points.jsonl")}
    states = {s["id"]: s["state"] for s in read(DATA / "states.jsonl")}
    labels = {lab["id"]: lab for lab in read(DATA / "labels" / "mammoth.jsonl")}
    out = DATA / "split"
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    files = {name: (out / f"{name}.jsonl").open("w") for name in ("train", "dev", "test")}
    for pid, p in points.items():
        if pid not in states or pid not in labels:
            continue
        if p["split"] != "train":
            assert not p["synthetic"], f"synthetic point {pid} in {p['split']}"
        files[p["split"]].write(json.dumps(example(p, states[pid], labels[pid]), ensure_ascii=False) + "\n")
        counts[p["split"]] += 1
        counts[f"{p['split']}_synthetic"] += bool(p["synthetic"])
    for f in files.values():
        f.close()
    print(json.dumps(dict(counts)), "->", out)


if __name__ == "__main__":
    main()
