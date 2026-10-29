"""Fit a softmax temperature on dev and store it next to the model.

    uv run python pipeline/calibrate.py runs/qwen3-1.7b/fused-8bit

The mod acts only above SPISEV_MIN_CONFIDENCE, so probabilities must mean what they say.
One scalar, fitted by minimizing the dev cross-entropy against Mammoth's distribution;
it never changes the pick. Reports the calibration error (ECE of the confidence against
agreement with Mammoth) before and after. From Nisev.
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import Classifier  # noqa: E402
from metrics import ece  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "split")
    args = parser.parse_args()
    from config import load_env
    load_env()

    clf = Classifier(args.model)
    clf.temperature = 1.0
    dev = [json.loads(line) for line in (args.split_dir / "dev.jsonl").open()]
    logits = mx.stack([clf.letter_logits(e["state"])[0] for e in dev])
    targets = mx.array([e["targets"] for e in dev], dtype=mx.float32)
    ref = mx.argmax(targets, axis=-1)

    def loss(t):
        logp = logits / t - mx.logsumexp(logits / t, axis=-1, keepdims=True)
        return -(targets * logp).sum(axis=-1).mean().item()

    def calibration(t):
        p = mx.softmax(logits / t, axis=-1)
        pick = mx.argmax(p, axis=-1)
        return ece(mx.max(p, axis=-1).tolist(), (pick == ref).tolist())

    grid = [0.5 + 0.05 * i for i in range(91)]
    best = min(grid, key=loss)
    result = {"temperature": round(best, 2), "dev_soft_ce": loss(best), "dev_soft_ce_uncalibrated": loss(1.0),
              "ece_before": calibration(1.0), "ece_after": calibration(best), "n": len(dev)}
    (args.model / "calibration.json").write_text(json.dumps(result, indent=1))
    print(f"temperature {best:.2f}: dev soft CE {loss(1.0):.3f} -> {loss(best):.3f}; "
          f"calibration error {result['ece_before']:.3f} -> {result['ece_after']:.3f}")


if __name__ == "__main__":
    main()
