"""What the pipeline produced, from local files only (no model call).

    uv run python pipeline/report.py

- counts per step, and spend (Haiku calls and list-price equivalent, Mammoth calls and
  input tokens, billed amounts);
- Mammoth's category distribution per split and a histogram of its confidence (the
  probability of its pick);
- agreement with weak labels: prompts after /clear and splices should be independent or
  related, short acknowledgements continuation. The headline is the precision of
  `independent`: of the weakly labeled points Mammoth calls independent, the share whose
  weak label says new task;
- floors on the test-candidate split: always `continuation`, and TF-IDF + logistic
  regression trained on the weak labels of the train split.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, load_env  # noqa: E402
from task import CATEGORIES  # noqa: E402

NEW_TASK = {"independent", "related"}


def read(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def weak_agreement(points, labels):
    rows = [(points[i]["weak"], labels[i]["choice"]) for i in labels if points.get(i, {}).get("weak")]
    out = {}
    for weak, ok in (("new_task", NEW_TASK), ("continuation", {"continuation"})):
        sub = [c for w, c in rows if w == weak]
        out[weak] = {"n": len(sub), "agree": round(sum(c in ok for c in sub) / len(sub), 3) if sub else None}
    out["independent_precision"] = independent_precision(rows)
    return out


def independent_precision(rows):
    picked = [w for w, c in rows if c == "independent"]
    return {"n": len(picked), "precision": round(sum(w == "new_task" for w in picked) / len(picked), 3)
            if picked else None}


def histogram(values, bins=10):
    counts = Counter(min(int(v * bins), bins - 1) for v in values)
    total = len(values) or 1
    return [(f"{b / bins:.1f}-{(b + 1) / bins:.1f}", counts[b], "#" * round(40 * counts[b] / total))
            for b in range(bins)]


def floors(points, labels, states):
    test = [i for i in labels if labels[i]["split"] == "test"]
    if not test:
        return {}
    always = {"agreement_with_mammoth": round(sum(labels[i]["choice"] == "continuation" for i in test) / len(test), 3),
              "independent_precision": {"n": 0, "precision": None}}
    train = [i for i in points if points[i]["split"] == "train" and points[i].get("weak") and i in states]
    out = {"n_test": len(test), "always_continuation": always}
    if len({points[i]["weak"] for i in train}) < 2:
        out["tfidf_weak"] = "not enough weak labels in train (need both kinds)"
        return out
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    msg = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=50_000)
    conv = TfidfVectorizer(min_df=1, sublinear_tf=True, max_features=20_000)

    def features(ids, fit=False):
        m = [states[i]["latest_user_message"] for i in ids]
        c = [states[i]["conversation"] for i in ids]
        if fit:
            return hstack([msg.fit_transform(m), conv.fit_transform(c)]).tocsr()
        return hstack([msg.transform(m), conv.transform(c)]).tocsr()

    y = ["independent" if points[i]["weak"] == "new_task" else "continuation" for i in train]
    clf = LogisticRegression(C=4, max_iter=5000, class_weight="balanced").fit(features(train, fit=True), y)
    ids = [i for i in test if i in states]
    pred = dict(zip(ids, clf.predict(features(ids))))
    rows = [(points[i]["weak"], pred[i]) for i in ids if points[i].get("weak")]
    out["tfidf_weak"] = {"train_weak_points": len(train),
                         "agreement_with_mammoth": round(sum(pred[i] == labels[i]["choice"] for i in ids) / len(ids), 3),
                         "independent_precision": independent_precision(rows)}
    return out


def main():
    load_env()
    sessions = sum(1 for _ in (DATA / "sessions.jsonl").open()) if (DATA / "sessions.jsonl").exists() else 0
    clean = json.loads((DATA / "clean-report.json").read_text()) if (DATA / "clean-report.json").exists() else {}
    build = json.loads((DATA / "build-report.json").read_text()) if (DATA / "build-report.json").exists() else {}
    points = {p["id"]: p for p in read(DATA / "points.jsonl")}
    states = {s["id"]: s["state"] for s in read(DATA / "states.jsonl")}
    labels = {lab["id"]: lab for lab in read(DATA / "labels" / "mammoth.jsonl") if lab["id"] in points}
    nodes = read(DATA / "tree-cache" / "nodes.jsonl")
    answers = read(DATA / "teacher-cache" / "answers.jsonl")

    print("== counts")
    print(f"extract: {sessions} sessions")
    print(f"clean: {clean.get('kept_sessions')} sessions kept, {clean.get('candidate_points')} candidate points; "
          f"dropped {clean.get('dropped_sessions')}")
    print(f"points: {build.get('points')} · splices {build.get('splices')} · linked to /clear "
          f"{build.get('linked_clear')} · sessions {build.get('sessions_per_split')}")
    per_split = Counter(points[i]["split"] for i in states if i in points)
    print(f"tree: {len(states)} states {dict(per_split)} · {len(nodes)} summary nodes in the cache")
    lab_split = Counter(lab["split"] for lab in labels.values())
    print(f"mammoth: {len(labels)} labeled {dict(lab_split)} · train points with a Mammoth label: "
          f"{lab_split['train']} (synthetic {sum(points[i]['synthetic'] == 'splice' for i in labels if labels[i]['split'] == 'train')})")

    print("\n== spend")
    calls = sum(n.get("calls", 1) for n in nodes)
    print(f"haiku: {calls} calls for {len(nodes)} nodes, list-price equivalent "
          f"${sum(n.get('list_usd', 0) for n in nodes):.2f}, billed $0.00 (subscription)")
    tokens = sum(a.get("input_tokens") or 0 for a in answers)
    print(f"mammoth: {len(answers)} calls, {tokens} input tokens, billed $0.00 (free week); versions "
          f"{dict(Counter(a.get('version') for a in answers))}")

    print("\n== Mammoth distribution")
    for split in ("train", "dev", "test"):
        c = Counter(lab["choice"] for lab in labels.values() if lab["split"] == split)
        n = sum(c.values())
        if n:
            print(f"{split:5} n={n:5} " + " · ".join(f"{k} {100 * c[k] / n:.1f}%" for k in CATEGORIES))

    print("\n== confidence (probability of the pick)")
    for rng, count, bar in histogram([max(lab["probabilities"].values()) for lab in labels.values()]):
        print(f"{rng} {count:5} {bar}")

    print("\n== weak labels")
    agreement = weak_agreement(points, labels)
    print(f"after /clear or spliced -> independent/related: {agreement['new_task']}")
    print(f"short acknowledgement -> continuation: {agreement['continuation']}")
    ip = agreement["independent_precision"]
    print(f"HEADLINE precision of independent (vs weak labels): {ip['precision']} (n={ip['n']})")

    print("\n== floors (test-candidates)")
    for name, value in floors(points, labels, states).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
