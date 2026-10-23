"""Bundle the training data and trainer for a run on a free Colab GPU.

    uv run python cloud/pack.py      # -> cloud/dist/spisev-bundle.tar.gz

States go through the shared clip (task.clip_state) and are tokenized here, with the
chat template call the local server uses (pipeline/infer.py), then shipped as token ids
with Mammoth's distribution as the soft target. The VM never renders a prompt, so
training sees exactly the tokens the served model will. Only data/split and the trainer
are packed: .env and the rest of the repo never leave this machine. From Nisev.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from task import LETTERS, clip_state, messages  # noqa: E402

SPLIT = ROOT / "data" / "split"
FILES = ("train_unsloth.py", "vm_packer.py", "soft_loss.py")


def encode(tokenizer, example):
    from infer import encode_prompt

    state = clip_state(example["state"]["conversation"], example["state"]["latest_user_message"])
    ids = encode_prompt(tokenizer, messages(state))
    answer = tokenizer.encode(example["completion"], add_special_tokens=False)
    assert len(answer) == 1, f"answer letter {example['completion']!r} is not one token"
    return {"id": example["id"], "category": example["category"], "prompt_ids": ids, "answer_id": answer[0],
            "targets": example["targets"]}


def main():
    from config import load_env
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Drop longer train examples")
    args = parser.parse_args()
    load_env()

    tokenizer = load_tokenizer(Path(snapshot_download(args.tokenizer, allow_patterns=["*.json", "*.jinja", "*.txt"])))
    stage = HERE / "dist" / "spisev"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data").mkdir(parents=True)
    counts = {}
    for name in ("train", "dev"):
        rows = [encode(tokenizer, json.loads(line)) for line in (SPLIT / f"{name}.jsonl").open()]
        if name == "train":
            rows = [r for r in rows if len(r["prompt_ids"]) + 1 <= args.max_tokens]
        with (stage / "data" / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[name] = len(rows)
    for script in FILES:
        shutil.copy(HERE / script, stage / script)
    lengths = sorted(len(json.loads(l)["prompt_ids"]) for l in (stage / "data" / "train.jsonl").open())
    manifest = {"tokenizer": args.tokenizer,
                "letter_ids": [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS],
                "im_end_id": tokenizer.encode("<|im_end|>", add_special_tokens=False)[0], "examples": counts,
                "prompt_tokens": {"p50": lengths[len(lengths) // 2], "p95": lengths[int(len(lengths) * 0.95)],
                                  "max": lengths[-1]}}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
    out = HERE / "dist" / "spisev-bundle.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(stage, arcname="spisev")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    print(json.dumps(manifest), f"\n{out} ({out.stat().st_size / 1e6:.1f} MB, sha256 {digest})")


if __name__ == "__main__":
    main()
