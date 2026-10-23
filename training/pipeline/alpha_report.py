"""The section 11 table: both alpha candidates and Mammoth on the held-out sessions.

    uv run python pipeline/alpha_report.py [--a qwen3-1.7b --b qwen3.5-2b --mammoth mammoth]

Reads results/NAME.json and the predictions files written by evaluate.py, and runs the
exact McNemar test between the two candidates on their per-point agreement with Mammoth.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def load(name):
    path = ROOT / "results" / f"{name}.json"
    if not path.exists():
        return None, None
    preds = {json.loads(line)["id"]: json.loads(line) for line in (ROOT / "results" / f"{name}.predictions.jsonl").open()}
    return json.loads(path.read_text()), preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="qwen3-1.7b")
    parser.add_argument("--b", default="qwen3.5-2b")
    parser.add_argument("--mammoth", default="mammoth")
    parser.add_argument("--labels", nargs=3, default=["Qwen3-1.7B (MLX LoRA), GGUF Q8_0",
                                                      "Qwen3.5-2B (Unsloth LoRA on Colab), GGUF Q8_0",
                                                      "Mammoth (Baseten, reference)"])
    args = parser.parse_args()
    print("| Model | Top-1 agreement with Mammoth | KL to Mammoth | Calibration error | Precision of `independent` "
          "(weak labels) | Action agreement | Latency p50 / p95 |")
    print("| :- | -: | -: | -: | -: | -: | -: |")
    for name, label in zip((args.a, args.b, args.mammoth), args.labels):
        r, _ = load(name)
        if not r:
            print(f"| {label} | (not run) | | | | | |")
            continue
        ref = name == args.mammoth
        lat = r["latency_ms"]
        print(f"| {label} | {'reference' if ref else pct(r['agreement'])} | {'0' if ref else f'{r['kl']:.3f}'} "
              f"| {'' if ref else f'{r['ece']:.3f}'} | {pct(r['independent_precision'])} (n={r['independent_n']}) "
              f"| {'reference' if ref else pct(r['action_agreement'])} | {lat['p50']:.0f} / {lat['p95']:.0f} ms |")
    (ra, pa), (rb, pb) = load(args.a), load(args.b)
    if pa and pb:
        ids = sorted(set(pa) & set(pb))
        p, b, c = mcnemar_exact([pa[i]["agrees"] for i in ids], [pb[i]["agrees"] for i in ids])
        print(f"\nExact McNemar test, {args.a} vs {args.b} on agreement with Mammoth (n={len(ids)}): "
              f"{b} points only {args.a} agrees, {c} only {args.b}; two-sided p = {p:.4f}")


if __name__ == "__main__":
    main()
