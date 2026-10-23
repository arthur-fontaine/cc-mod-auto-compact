"""Score a saved checkpoint on dev as one epoch, then pick the run's best epoch again. Runs on the VM.

    python eval_checkpoint.py --base unsloth/Qwen3-1.7B --run /root/runs/qwen3-1.7b --checkpoint checkpoint-450 --epoch 1

For a run resumed exactly at an epoch boundary: the trainer saved that checkpoint and then
failed in its end-of-epoch evaluation, so the resumed run never scores that epoch. This
does what train_unsloth.EpochEval would have (dev agreement with Mammoth and soft CE,
the adapter saved as epoch-N), then reselects `best` and summary.json over all epochs.
"""
import argparse
import json
import shutil
from pathlib import Path

from unsloth import FastLanguageModel  # Unsloth must be imported before transformers

from train_unsloth import dev_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--max-len", type=int, default=12288)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.run / "checkpoints" / args.checkpoint), max_seq_length=args.max_len,
        load_in_4bit=False, load_in_16bit=True)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCDE"]
    dev = [json.loads(line) for line in (args.data / "dev.jsonl").open()]
    agreement, ce = dev_scores(model, dev, letters)
    model.save_pretrained(args.run / f"epoch-{args.epoch}")
    step = json.loads((args.run / "checkpoints" / args.checkpoint / "trainer_state.json").read_text())["global_step"]
    metrics_path = args.run / "dev_metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.open()] if metrics_path.exists() else []
    metrics = sorted([m for m in metrics if m["epoch"] != args.epoch] +
                     [{"epoch": args.epoch, "step": step, "dev_agreement": agreement, "dev_soft_ce": ce}],
                     key=lambda m: m["epoch"])
    metrics_path.write_text("".join(json.dumps(m) + "\n" for m in metrics))
    best = max(metrics, key=lambda m: (m["dev_agreement"], -m["dev_soft_ce"]))
    if (args.run / "best").exists():
        shutil.rmtree(args.run / "best")
    shutil.copytree(args.run / f"epoch-{best['epoch']}", args.run / "best")
    tok.save_pretrained(args.run / "best")
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"base": args.base}
    summary.update(best=best, epochs=metrics)
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"epoch {args.epoch}: dev agreement {agreement:.3f}, soft CE {ce:.3f}; best: epoch {best['epoch']}")


if __name__ == "__main__":
    main()
