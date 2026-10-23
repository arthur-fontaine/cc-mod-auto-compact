"""Pick a run's best epoch on the dev split and make it the run's adapter.

    uv run python pipeline/select_checkpoint.py runs/qwen3-1.7b

Scores every epoch checkpoint (NNNNNNN_adapters.safetensors) and the final adapter on
dev by agreement with Mammoth, ties broken by the lower KL to its distribution. The
winner is copied to adapters.safetensors and the final one kept as
adapters-final.safetensors. The test split is never read here. From Nisev.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import Classifier  # noqa: E402
from metrics import score  # noqa: E402
from task import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def dev_rows():
    return [json.loads(line) for line in (ROOT / "data" / "split" / "dev.jsonl").open()]


def evaluate(clf, rows):
    out = []
    for e in rows:
        out.append({"probs": clf.predict(e["state"])["probabilities"],
                    "reference": dict(zip(CATEGORIES, e["targets"])), "weak": e.get("weak")})
    s = score(out)
    return s["agreement"], s["kl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    from config import load_env
    load_env()

    base = json.loads((args.run / "adapter_config.json").read_text())["model"]
    rows = dev_rows()
    final = args.run / "adapters-final.safetensors"
    if not final.exists():
        shutil.copy(args.run / "adapters.safetensors", final)
    results = {}
    for weights in sorted(args.run.glob("*_adapters.safetensors")) + [final]:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(args.run / "adapter_config.json", tmp)
            shutil.copy(weights, Path(tmp) / "adapters.safetensors")
            agreement, kl = evaluate(Classifier(base, tmp), rows)
        results[weights.name] = {"agreement": agreement, "kl": kl}
        print(f"{weights.name}: dev agreement {agreement:.3f} · KL {kl:.3f}", flush=True)
    best = max(results, key=lambda k: (results[k]["agreement"], -results[k]["kl"]))
    shutil.copy(args.run / best, args.run / "adapters.safetensors")
    (args.run / "selection.json").write_text(json.dumps({"best": best, "dev": results, "n": len(rows)}, indent=1))
    print(f"best: {best}")


if __name__ == "__main__":
    main()
