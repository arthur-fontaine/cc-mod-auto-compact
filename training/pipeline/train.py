"""LoRA-distill Mammoth into a dense Qwen3 model with MLX, on Apple silicon.

    uv run python pipeline/train.py --base Qwen/Qwen3-1.7B --out runs/qwen3-1.7b --epochs 4
    uv run python pipeline/train.py ... --hard-targets       # Mammoth's pick as a one-hot target

Nisev's recipe (pipeline/train.py there): LoRA rank 16 / alpha 32 on every linear layer
of every block, loss on the answer position only, AdamW 1e-4 with warmup and cosine
decay, gradients clipped to norm 1, an adapter saved at each epoch end
(select_checkpoint.py picks the best on dev). The target is Mammoth's whole
distribution over the option letters: soft cross-entropy at the answer position, which
is the standard cross-entropy when the target is one-hot (tests/test_losses.py).

Prompts are rendered from states clipped by task.clip_state, the clip shared with the
packer and the mod, with the base model's chat template in non-thinking mode, exactly
as the server renders them.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import CATEGORIES, LETTERS, clip_state, messages  # noqa: E402


def soft_cross_entropy(logits, letter_ids, targets):
    """-sum_c t_c log softmax(logits)[letter_c], over the whole vocabulary, per example.

    logits (B, V), letter_ids (C,), targets (B, C) summing to 1.
    """
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(targets * logp[:, letter_ids]).sum(axis=-1)


def encode(tokenizer, example, max_len, letter_ids, hard=False):
    from infer import encode_prompt

    state = clip_state(example["state"]["conversation"], example["state"]["latest_user_message"])
    prompt = encode_prompt(tokenizer, messages(state))
    answer = tokenizer.encode(example["completion"], add_special_tokens=False)
    if len(prompt) + 1 > max_len:
        return None  # Truncating would cut the answer position.
    targets = [0.0] * len(CATEGORIES)
    if hard:
        targets[LETTERS.index(example["completion"])] = 1.0
    else:
        total = sum(example["targets"])
        targets = [t / total for t in example["targets"]]
    return list(prompt) + answer, len(prompt), targets


def make_loss(letter_ids):
    letters = mx.array(letter_ids)

    def loss(model, batch, lengths, targets):
        """The LM head runs on the answer position only (lengths[:, 0]), not on the ~150k-token
        vocabulary at every position."""
        lm = getattr(model, "language_model", model)
        hidden = lm.model(batch[:, :-1])
        pos = (lengths[:, 0] - 1)[:, None]
        h = mx.take_along_axis(hidden, mx.broadcast_to(pos[..., None], (pos.shape[0], 1, hidden.shape[-1])), axis=1)
        logits = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
        ce = soft_cross_entropy(logits[:, 0].astype(mx.float32), letters, targets)
        return ce.sum() / ce.size, mx.array(ce.size)

    return loss


def iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    """mlx-lm's batching (length-sorted batches, random order) with the soft targets alongside."""
    idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
    batches = [idx[i:i + batch_size] for i in range(0, len(idx) - batch_size + 1, batch_size)]
    rng = np.random.default_rng(seed)
    while True:
        for b in rng.permutation(len(batches)):
            items = [dataset[j] for j in batches[b]]
            lengths = [min(len(t), max_seq_length) for t, _, _ in items]
            width = min(1 + 32 * ((max(lengths) + 31) // 32), max_seq_length)
            arr = np.zeros((len(items), width), np.int32)
            for k, (t, _, _) in enumerate(items):
                arr[k, :lengths[k]] = t[:lengths[k]]
            yield (mx.array(arr), mx.array([[o, n] for (_, o, _), n in zip(items, lengths)]),
                   mx.array([tg for _, _, tg in items], dtype=mx.float32))
        if not loop:
            break


class ClippedAdamW(optim.AdamW):
    """AdamW with global gradient-norm clipping, as in tev1 (max_grad_norm 1)."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def update(self, model, gradients):
        gradients, _ = optim.clip_grad_norm(gradients, self.max_norm)
        return super().update(model, gradients)


class Dataset:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)


def main():
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hard-targets", action="store_true", help="Train on Mammoth's pick, one-hot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from config import load_env
    load_env()
    mx.random.seed(args.seed)
    mx.set_cache_limit(2 * 1024**3)
    model, tokenizer = load(args.base)
    letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    read = lambda name: [json.loads(line) for line in (args.data / f"{name}.jsonl").open()]
    enc = lambda rows: [x for x in (encode(tokenizer, ex, args.max_seq, letter_ids, args.hard_targets)
                                    for ex in rows) if x]
    train_set, dev_set = Dataset(enc(read("train"))), Dataset(enc(read("dev")))
    lengths = [len(t) for t, _, _ in train_set.items]
    print(f"train {len(train_set)} · dev {len(dev_set)} · tokens/example p50 {int(np.median(lengths))} "
          f"p95 {int(np.percentile(lengths, 95))} max {max(lengths)}", flush=True)

    model.freeze()
    lora = {"rank": args.rank, "scale": args.alpha / args.rank, "dropout": 0.0}
    linear_to_lora_layers(model, len(model.layers), lora)
    print_trainable_parameters(model)

    per_epoch = len(train_set) // args.batch_size
    iters = per_epoch * args.epochs
    updates = math.ceil(iters / args.grad_accum)
    warmup = max(1, int(updates * args.warmup))
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.lr, warmup), optim.cosine_decay(args.lr, updates - warmup, 0.0)], [warmup])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adapter_config.json").write_text(json.dumps({
        "model": args.base, "fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora,
        "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "iters": iters, "iters_per_epoch": per_epoch, "train_examples": len(train_set),
    }, indent=1))
    started = time.time()
    train(model=model,
          optimizer=ClippedAdamW(learning_rate=schedule, weight_decay=0.0, max_norm=args.max_grad_norm),
          train_dataset=train_set, val_dataset=dev_set, loss=make_loss(letter_ids), iterate_batches=iterate_batches,
          args=TrainingArgs(batch_size=args.batch_size, iters=iters, val_batches=-1, steps_per_report=20,
                            steps_per_eval=per_epoch, steps_per_save=per_epoch, max_seq_length=args.max_seq,
                            adapter_file=str(args.out / "adapters.safetensors"), grad_checkpoint=True,
                            grad_accumulation_steps=args.grad_accum))
    print(f"trained {iters} iters ({args.epochs} epochs) in {(time.time() - started) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
