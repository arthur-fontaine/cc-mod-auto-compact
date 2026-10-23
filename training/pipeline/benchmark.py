"""Benchmark the decision models a user can plug into the mod, on an answer key none of them wrote.

    uv run python pipeline/hindsight.py ...                 # the answer key (see its docstring)
    uv run python pipeline/evaluate.py --endpoint ... --name NAME [--split test-splices --name NAME-splices]
    uv run python pipeline/benchmark.py                     # -> results/bench-*.json and the table
    uv run python pipeline/plot.py                          # -> ../docs/benchmark.svg

The points are the held-out test prompts and the held-out splices (a session's end followed
by another session's first prompt), labeled with hindsight by Sonnet 5.5 and Opus 5.5; the
key keeps the points where both teachers lead to the same action. Each model's category is
mapped to the action the mod would take (pipeline/policy.py). Predictions come from
evaluate.py, one request at a time, so latencies are comparable; a request that failed is
scored as keep, which is what the mod does.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import mcnemar_exact  # noqa: E402
from policy import action  # noqa: E402
from task import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
AUTO_CONFIDENCE = 0.85
TIMEOUT_MS = 4000  # the mod's default: past it, it keeps the context

# name, label, where it runs, predictions on test, predictions on held-out splices
CONTENDERS = [
    ("spisev", "Spisev 1.7B", "Local · llama.cpp · Q8_0, 1.8 GB", "qwen3-1.7b", "qwen3-1.7b-splices"),
    ("qwen3.5-2b", "Qwen3.5-2B (same recipe)", "Local · llama.cpp · Q8_0, 2.0 GB", "qwen3.5-2b", "qwen3.5-2b-splices"),
    ("kev-4b", "Kev-4B (zero-shot)", "Local · llama.cpp · Q8_0, 4.5 GB", "kev-4b", "kev-4b-splices"),
    ("elephant", "Elephant 9B", "Cloud · Baseten", "elephant", "elephant-splices"),
    ("mammoth", "Mammoth 27B", "Cloud · Baseten", "mammoth", None),
]


def read(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def gold():
    return {g["id"]: g for g in read(DATA / "labels" / "hindsight-gold.jsonl") if g["action"]}


def predictions(test_name, splice_name):
    """{id: (probabilities or None, ms or None)} over test and splices."""
    out = {}
    for name in (test_name, splice_name):
        if name:
            for r in read(RESULTS / f"{name}.predictions.jsonl"):
                # Latency from the test runs only: each ran alone on the machine, one request at a time.
                out[r["id"]] = (r["probabilities"], r.get("ms") if name == test_name else None)
    if test_name == "mammoth":  # its splice answers are the teacher labels, asked the same way
        for r in read(DATA / "labels" / "mammoth-test-splices.jsonl"):
            out.setdefault(r["id"], (r["probabilities"], None))
    return out


def decide(probs):
    if probs is None:
        return None, "keep", 0.0
    pick = max(CATEGORIES, key=lambda c: probs.get(c, 0.0))
    return pick, action(pick), probs[pick]


def baseline_accuracy(g):
    return sum(x["action"] == "keep" for x in g.values()) / len(g)


def threshold(n, p0, alpha=0.05):
    """The smallest accuracy a one-sided exact binomial test puts above always keeping, at alpha."""
    tail = lambda k: sum(math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1))
    return next(k for k in range(n + 1) if tail(k) < alpha) / n


def score(preds, g):
    rows = []
    for pid, gl in g.items():
        if pid not in preds:
            continue
        probs, ms = preds[pid]
        cat, act, conf = decide(probs)
        late = ms is not None and ms > TIMEOUT_MS
        applied = act if act == "keep" or (conf >= AUTO_CONFIDENCE and not late) else "keep"
        rows.append({"id": pid, "gold_action": gl["action"], "gold_category": gl["category"], "action": act,
                     "category": cat, "applied": applied, "ms": ms, "error": probs is None})
    n = len(rows)
    keep = [r for r in rows if r["gold_action"] == "keep"]
    act = [r for r in rows if r["gold_action"] != "keep"]
    cats = [r for r in rows if r["gold_category"]]
    ms = sorted(r["ms"] for r in rows if r["ms"] is not None)
    return {
        "n": n,
        "right_action": sum(r["action"] == r["gold_action"] for r in rows) / n,
        "kept_when_it_should": sum(r["action"] == "keep" for r in keep) / len(keep),
        "acted_when_it_should": sum(r["action"] == r["gold_action"] for r in act) / len(act),
        "right_action_applied": sum(r["applied"] == r["gold_action"] for r in rows) / n,
        "right_category": sum(r["category"] == r["gold_category"] for r in cats) / len(cats),
        "errors": sum(r["error"] for r in rows),
        "timeouts": sum(r["ms"] is not None and r["ms"] > TIMEOUT_MS for r in rows),
        "latency_ms": {"p50": ms[len(ms) // 2], "p95": ms[min(len(ms) - 1, int(len(ms) * 0.95))]} if ms else None,
        "rows": rows,
    }


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    g = gold()
    base = baseline_accuracy(g)
    bar = threshold(len(g), base)
    results = {}
    for name, label, where, test_name, splice_name in CONTENDERS:
        preds = predictions(test_name, splice_name)
        if not preds:
            print(f"{name}: no predictions, skipped", file=sys.stderr)
            continue
        s = score(preds, g)
        results[name] = s
        out = {k: v for k, v in s.items() if k != "rows"}
        (RESULTS / f"bench-{name}.json").write_text(json.dumps(
            {"name": name, "label": label, "where": where, **out, "baseline_keep": base, "threshold": bar}, indent=1))
    keep_only = {pid: (None, None) for pid in g}
    print(f"Answer key: {len(g)} points (both teachers agree on the action); always keep: {pct(base)}; "
          f"beats it at p < 0.05 from {pct(bar)}\n")
    print("| Model | Runs | Right action | Kept when it should | Acted when it should | Right action, auto mode "
          "| Right category | Latency p50 / p95 |")
    print("| :- | :- | -: | -: | -: | -: | -: | -: |")
    for name, label, where, *_ in CONTENDERS:
        if name not in results:
            continue
        s = results[name]
        lat = s["latency_ms"]
        latency = f"{lat['p50']:.0f} / {lat['p95']:.0f} ms" if lat else ""
        err = f" ({s['errors']} errors, kept)" if s["errors"] else ""
        print(f"| {label} | {where} | {pct(s['right_action'])}{err} | {pct(s['kept_when_it_should'])} "
              f"| {pct(s['acted_when_it_should'])} | {pct(s['right_action_applied'])} | {pct(s['right_category'])} "
              f"| {latency} |")
    base_s = score(keep_only, g)
    print(f"| Always keep | | {pct(base_s['right_action'])} | 100.0% | 0.0% | {pct(base_s['right_action'])} "
          f"| | |")
    print("\nExact McNemar against always keep, on right action (one-sided where the model is better):")
    for name in results:
        rows = results[name]["rows"]
        mine = [r["action"] == r["gold_action"] for r in rows]
        always = [r["gold_action"] == "keep" for r in rows]
        p, b, c = mcnemar_exact(mine, always)
        print(f"  {name}: {b} points only it gets right, {c} only always-keep; p = {p / 2 if b > c else 1 - p / 2:.4f}")
    if "spisev" in results:
        print("\nExact McNemar, Spisev against each model, on right action (two-sided):")
        mine = {r["id"]: r["action"] == r["gold_action"] for r in results["spisev"]["rows"]}
        for name in results:
            if name == "spisev":
                continue
            other = {r["id"]: r["action"] == r["gold_action"] for r in results[name]["rows"]}
            ids = sorted(set(mine) & set(other))
            p, b, c = mcnemar_exact([mine[i] for i in ids], [other[i] for i in ids])
            print(f"  against {name}: {b} only Spisev gets right, {c} only {name}; p = {p:.4f}")


if __name__ == "__main__":
    main()
