"""Where Mammoth labeling stands, from the local files only: no API call.

    uv run python scripts/teacher_status.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from config import DATA, load_env  # noqa: E402
from teacher import CACHE, LOG, STATE, cache_key  # noqa: E402


def main():
    load_env()
    import os

    from task import request

    model = os.environ.get("DECISION_MODEL") or ""
    cached = set()
    path = CACHE / "answers.jsonl"
    if path.exists():
        cached = {json.loads(line)["key"] for line in path.open()}
    states = DATA / "states.jsonl"
    splits, labeled = Counter(), Counter()
    if states.exists():
        for line in states.open():
            r = json.loads(line)
            splits[r["split"]] += 1
            labeled[r["split"]] += cache_key(model, request(r["state"], model)) in cached
    events = [json.loads(line) for line in LOG.open()] if LOG.exists() else []
    errors = [e for e in events if e.get("status") != 200]
    print(f"model: {model or '(DECISION_MODEL unset)'} · served version: "
          f"{json.loads(STATE.read_text()).get('version') if STATE.exists() else 'none yet'}")
    print(f"cached answers: {len(cached)}")
    for split in ("train", "dev", "test"):
        if splits[split]:
            print(f"  {split}: {labeled[split]} labeled / {splits[split] - labeled[split]} remaining")
    print(f"labeled: {sum(labeled.values())} · remaining: {sum(splits.values()) - sum(labeled.values())}")
    print(f"429s: {sum(e.get('status') == 429 for e in events)}")
    if errors:
        last = errors[-1]
        print(f"last error: HTTP {last.get('status')} ({last.get('decision')}): {last.get('body', '')[:300]}")
    else:
        print("last error: none")


if __name__ == "__main__":
    main()
