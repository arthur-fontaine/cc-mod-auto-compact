"""Merge a LoRA adapter trained on Colab into its base model and convert it to MLX.

    uv run --group merge python cloud/import_adapter.py runs/cloud/qwen3.5-2b --base Qwen/Qwen3.5-2B

Reads RUN/synced/best (the PEFT adapter train_unsloth.py picked on dev), writes RUN/merged
(Hugging Face, bf16; what export_gguf.py converts) and RUN/fused-8bit (MLX, what
evaluate.py and calibrate.py load). It stops if the adapter attaches to no layer, and it
refuses to export a model whose dev agreement with Mammoth moved more than 3 points from
the one measured during training.

Unsloth's fused MLP LoRA path trains the MLP adapters at half the scaling the adapter
declares, so they are applied at half scale here (Nisev measured 54.4% on dev loaded as
declared against 60.6% at half scale, 61.9% in training). From Nisev.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from soft_loss import answer_logits  # noqa: E402


@torch.no_grad()
def dev_agreement(model, device, bundle="spisev"):
    data = ROOT / "cloud" / "dist" / bundle
    letters = json.loads((data / "manifest.json").read_text())["letter_ids"]
    rows = [json.loads(line) for line in (data / "data" / "dev.jsonl").open()]
    agree = 0
    for r in rows:
        ids = torch.tensor([r["prompt_ids"]], device=device)
        logits = answer_logits(model, ids, torch.ones_like(ids), torch.tensor([len(r["prompt_ids"]) - 1]))
        agree += int(logits[0, letters].argmax().item() == max(range(len(r["targets"])), key=r["targets"].__getitem__))
    return agree / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--mlp-scale", type=float, default=0.5, help="Factor on the MLP adapters' declared scaling")
    parser.add_argument("--max-gap", type=float, default=0.03, help="Allowed dev agreement gap with training")
    parser.add_argument("--bundle", default="spisev", help="The cloud/dist bundle the run trained on")
    parser.add_argument("--skip-mlx", action="store_true", help="Stop after the merge (on a GPU host without MLX)")
    args = parser.parse_args()
    from config import load_env
    load_env()

    adapter = args.run / "synced" / "best"
    merged_dir, mlx_dir = args.run / "merged", args.run / f"fused-{args.bits}bit"
    base_path = snapshot_download(args.base)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    multimodal = "vision_config" in AutoConfig.from_pretrained(base_path).to_dict()
    auto = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = auto.from_pretrained(base_path, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, adapter).eval()
    layers = [(name, m) for name, m in model.named_modules() if isinstance(getattr(m, "scaling", None), dict)]
    kinds = sorted({name.split(".")[-1] for name, _ in layers})
    print(f"LoRA attached to {len(layers)} modules: {kinds}")
    if not layers:
        sys.exit("The adapter attached to no module: module names differ between the cloud and local model.")
    for name, m in layers:
        if ".mlp." in name:
            m.scaling = {k: v * args.mlp_scale for k, v in m.scaling.items()}

    trained = json.loads((args.run / "synced" / "summary.json").read_text())["best"]["dev_agreement"]
    measured = dev_agreement(model, device, args.bundle)
    print(f"dev agreement with Mammoth: {measured:.3f} here, {trained:.3f} in training")
    if abs(measured - trained) > args.max_gap:
        sys.exit("The merged model does not reproduce the training-time dev agreement; not exporting it.")
    print("dev check passed")

    model = model.merge_and_unload().to("cpu")
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    model.save_pretrained(merged_dir)
    (AutoProcessor if multimodal else AutoTokenizer).from_pretrained(base_path).save_pretrained(merged_dir)
    if (Path(base_path) / "chat_template.jinja").exists():
        shutil.copy(Path(base_path) / "chat_template.jinja", merged_dir / "chat_template.jinja")
    if args.skip_mlx:
        print(f"-> {merged_dir}")
        return
    if mlx_dir.exists():
        shutil.rmtree(mlx_dir)
    convert = Path(sys.executable).parent / "mlx_lm.convert"
    subprocess.run([str(convert), "--hf-path", str(merged_dir), "--mlx-path", str(mlx_dir), "-q", "--q-bits",
                    str(args.bits)], check=True)
    print(f"-> {mlx_dir}")


if __name__ == "__main__":
    main()
