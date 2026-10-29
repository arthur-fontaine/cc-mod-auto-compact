"""Score state formats on the dev hindsight key, to choose one without touching test (PLAN.md 3).

    uv run python pipeline/choose_format.py mammoth-v1-dev mammoth-v2-128-dev mammoth-v2-192-dev ...
    uv run python pipeline/choose_format.py --neutral ...   # the key whose teachers read no summary line

Each name is a predictions file in results/ (evaluate.py). The first is the reference: every
other is compared with it point by point (exact McNemar on right action and on right category).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark  # noqa: E402
from hindsight import paths  # noqa: E402
from metrics import mcnemar_exact  # noqa: E402


def dev_gold(neutral=False):
    return {g["id"]: g for g in map(json.loads, paths("dev", neutral)[1].open()) if g["action"]}


def predictions(name):
    return {r["id"]: (r["probabilities"], r.get("ms")) for r in
            map(json.loads, (benchmark.RESULTS / f"{name}.predictions.jsonl").open())}


def compare(rows, ref, field):
    right = lambda r, k: r[k] == r["gold_" + k]
    mine = {r["id"]: right(r, field) for r in rows if field == "action" or r["gold_category"]}
    base = {r["id"]: right(r, field) for r in ref if field == "action" or r["gold_category"]}
    ids = sorted(set(mine) & set(base))
    return mcnemar_exact([mine[i] for i in ids], [base[i] for i in ids])


def main(names):
    neutral = "--neutral" in names
    names = [n for n in names if n != "--neutral"]
    g = dev_gold(neutral)
    scored = {n: benchmark.score(predictions(n), g) for n in names}
    ref = scored[names[0]]["rows"]
    print(f"dev key: {len(g)} points; always keep {benchmark.pct(benchmark.baseline_accuracy(g))}\n")
    print("| Format | n | Right action | Acted when it should | Right category | vs first: action b/c p | category b/c p |")
    print("| :- | -: | -: | -: | -: | -: | -: |")
    out = {}
    for n, s in scored.items():
        pa, ba, ca = compare(s["rows"], ref, "action")
        pc, bc, cc = compare(s["rows"], ref, "category")
        print(f"| {n} | {s['n']} | {benchmark.pct(s['right_action'])} | {benchmark.pct(s['acted_when_it_should'])} "
              f"| {benchmark.pct(s['right_category'])} | {ba}/{ca} p={pa:.3f} | {bc}/{cc} p={pc:.3f} |")
        out[n] = {k: v for k, v in s.items() if k != "rows"}
    (benchmark.RESULTS / "choose-format.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
