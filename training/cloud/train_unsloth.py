"""Distill Mammoth into Qwen3.5 with Unsloth, on one CUDA GPU (a free Colab T4 works).

    python train_unsloth.py --base unsloth/Qwen3.5-2B --out /content/runs/qwen3.5-2b

Same recipe as pipeline/train.py: LoRA rank 16 / alpha 32 on EVERY linear layer of the
text model (soft_loss.QWEN35_TARGETS, linear-attention projections included), soft
cross-entropy at the answer position against Mammoth's distribution (--hard-targets
for its pick alone), AdamW 1e-4, cosine decay with 5% warmup, gradients clipped to 1,
4 epochs, best epoch on dev (agreement with Mammoth).

Resumable: checkpoints every --save-steps optimizer steps to OUT/checkpoints; rerun the
same command after a disconnect and it continues from the latest. Each epoch end writes
the dev metrics to OUT/dev_metrics.jsonl and the adapter to OUT/epoch-N; the best epoch
is copied to OUT/best. From Nisev's train_unsloth.py.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from unsloth import FastLanguageModel, is_bf16_supported  # Unsloth must be imported before transformers

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from soft_loss import QWEN35_TARGETS, answer_logits, soft_cross_entropy


class Examples(torch.utils.data.Dataset):
    def __init__(self, rows, hard):
        self.rows, self.hard = rows, hard

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        targets = list(r["targets"])
        if self.hard:
            best = max(range(len(targets)), key=targets.__getitem__)
            targets = [float(k == best) for k in range(len(targets))]
        total = sum(targets)
        return {"input_ids": r["prompt_ids"], "position": len(r["prompt_ids"]) - 1,
                "targets": [t / total for t in targets]}


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            # Right padding: the model is causal, so padding after the prompt changes nothing before it.
            ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[i, :len(b["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "position": torch.tensor([b["position"] for b in batch]),
                "targets": torch.tensor([b["targets"] for b in batch], dtype=torch.float32)}
    return fn


class SoftTrainer(Trainer):
    def __init__(self, *args, letter_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.letter_ids = letter_ids

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = answer_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["position"])
        letters = torch.tensor(self.letter_ids, device=logits.device)
        loss = soft_cross_entropy(logits, letters, inputs["targets"].to(logits.device)).mean()
        return (loss, {"logits": logits}) if return_outputs else loss


@torch.no_grad()
def dev_scores(model, rows, letter_ids):
    was_training = model.training
    model.eval()
    letters = torch.tensor(letter_ids, device=model.device)
    agree, ce = 0, 0.0
    for r in rows:
        ids = torch.tensor([r["prompt_ids"]], device=model.device)
        logits = answer_logits(model, ids, torch.ones_like(ids), torch.tensor([len(r["prompt_ids"]) - 1]))
        targets = torch.tensor([r["targets"]], device=model.device, dtype=torch.float32)
        targets = targets / targets.sum()
        agree += int(logits[0, letters].argmax().item() == targets[0].argmax().item())
        ce += soft_cross_entropy(logits, letters, targets).item()
    if was_training:
        model.train()
    return agree / len(rows), ce / len(rows)


class EpochEval(TrainerCallback):
    def __init__(self, out, model, dev, letter_ids):
        self.out, self.model, self.dev, self.letter_ids = out, model, dev, letter_ids

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        agreement, ce = dev_scores(self.model, self.dev, self.letter_ids)
        self.model.save_pretrained(self.out / f"epoch-{epoch}")
        with (self.out / "dev_metrics.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "step": state.global_step, "dev_agreement": agreement,
                                "dev_soft_ce": ce}) + "\n")
        print(f"epoch {epoch}: dev agreement with Mammoth {agreement:.3f}, soft CE {ce:.3f}", flush=True)


def latest_checkpoint(directory):
    found = sorted(directory.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(found[-1]) if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--save-steps", type=int, default=50, help="Optimizer steps between checkpoints")
    parser.add_argument("--targets", default=",".join(QWEN35_TARGETS))
    parser.add_argument("--hard-targets", action="store_true")
    args = parser.parse_args()
    targets = args.targets.split(",")
    missing = set(QWEN35_TARGETS) - set(targets)
    if missing:
        raise SystemExit(f"--targets must list every linear layer; missing {sorted(missing)}")

    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    read = lambda name: [json.loads(line) for line in (args.data / f"{name}.jsonl").open()]
    train_rows = [r for r in read("train") if len(r["prompt_ids"]) + 1 <= args.max_len]
    dev_rows = read("dev")
    args.out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_len, load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCDE"]
    assert letters == manifest["letter_ids"], f"tokenizer mismatch: {letters} vs {manifest['letter_ids']}"
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none", target_modules=targets,
        use_gradient_checkpointing="unsloth", random_state=42, max_seq_length=args.max_len)
    model.print_trainable_parameters()
    adapted = {n.split(".lora_A")[0].split(".")[-1] for n, _ in model.named_parameters() if "lora_A" in n}
    print(f"adapted layer names: {sorted(adapted)}", flush=True)

    trainer = SoftTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"), per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0, max_grad_norm=1.0,
            fp16=not is_bf16_supported(), bf16=is_bf16_supported(), logging_steps=10, save_strategy="steps",
            save_steps=args.save_steps, save_total_limit=2, train_sampling_strategy="group_by_length",
            remove_unused_columns=False, report_to="none", seed=42),
        train_dataset=Examples(train_rows, args.hard_targets),
        data_collator=collate(tok.pad_token_id if tok.pad_token_id is not None else manifest["im_end_id"]),
        callbacks=[EpochEval(args.out, model, dev_rows, letters)],
        letter_ids=letters,
    )
    resume = latest_checkpoint(args.out / "checkpoints")
    print(f"train {len(train_rows)} · dev {len(dev_rows)} · resume from {resume or 'scratch'}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    metrics = [json.loads(line) for line in (args.out / "dev_metrics.jsonl").open()]
    best = max(metrics, key=lambda m: (m["dev_agreement"], -m["dev_soft_ce"]))
    shutil.copytree(args.out / f"epoch-{best['epoch']}", args.out / "best", dirs_exist_ok=True)
    tok.save_pretrained(args.out / "best")
    (args.out / "summary.json").write_text(json.dumps({"base": args.base, "best": best, "epochs": metrics,
                                                         "targets": targets,
                                                         "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    print(f"best: epoch {best['epoch']} (dev agreement {best['dev_agreement']:.3f}) -> {args.out / 'best'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
