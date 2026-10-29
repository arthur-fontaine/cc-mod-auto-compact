"""Export spisev-alpha as a llama.cpp decision model (GGUF), served at /v1/systemone.

    uv run --group merge python pipeline/export_gguf.py runs/qwen3-1.7b/fused --calibrated runs/qwen3-1.7b/fused-8bit \
        --llama-cpp external/llama.cpp --name spisev-alpha-1.7b      # -> runs/qwen3-1.7b/spisev-alpha-1.7b-Q8_0.gguf

llama.cpp serves "decision models" from GGUF metadata. This one uses the `openjev`
decision type: one letter per option, read from the next-token logits after a prompt
that a `systemone` chat template renders from the request. The template rebuilds exactly
the prompt the model was trained on (pipeline/task.py): the system prompt, then the JSON
task with Python's separators, the request's question and options, and the state fields
(v1: `conversation`, `latest_user_message`; `--format v2`: `memory`, `last_agent_message`,
`latest_user_message`). llama.cpp's `tojson` keeps non-ASCII as
is, like json.dumps(..., ensure_ascii=False). The softmax temperature fitted on dev
(pipeline/calibrate.py) is stored as `<arch>.decision.temperature.choice`.

Needs a llama.cpp source tree for convert_hf_to_gguf.py and gguf-py (clone it into the
gitignored external/). From Nisev's export_gguf.py.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import LETTERS, SYSTEM  # noqa: E402

ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


FIELDS = {"v1": ("conversation", "latest_user_message"),
          "v2": ("memory", "last_agent_message", "latest_user_message")}


def systemone_template(fmt="v1"):
    option = ('{"label": "{{ letters[loop.index0] }}", "key": {{ o.key | tojson }}, '
              '"description": {{ o.description | tojson }}}')
    return (
        "{% set letters = '" + LETTERS + "' %}"
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n"
        '{"question": {{ instructions | tojson }}, "options": ['
        "{% for o in options %}{% if not loop.first %}, {% endif %}" + option + "{% endfor %}"
        '], "state": {' + ", ".join(f'"{k}": {{{{ state.{k} | tojson }}}}' for k in FIELDS[fmt]) + '}}<|im_end|>\n'
        + ASSISTANT_PREFIX
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="A Hugging Face checkpoint (the run's fused or merged directory)")
    parser.add_argument("--calibrated", type=Path, help="Directory holding calibration.json")
    parser.add_argument("--llama-cpp", type=Path, default=ROOT / "external" / "llama.cpp")
    parser.add_argument("--outtype", default="q8_0")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--name", default="spisev-alpha")
    parser.add_argument("--format", choices=sorted(FIELDS), default="v1", help="The state the model was trained on")
    args = parser.parse_args()

    temperature = args.temperature or json.loads((args.calibrated / "calibration.json").read_text())["temperature"]
    raw = args.model.parent / f"{args.name}.{args.outtype}.raw.gguf"
    out = args.model.parent / f"{args.name}-{args.outtype.upper()}.gguf"
    subprocess.run([sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(args.model),
                    "--outtype", args.outtype, "--outfile", str(raw)], check=True)

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    reader = gguf.GGUFReader(raw, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    string, f32 = gguf.GGUFValueType.STRING, gguf.GGUFValueType.FLOAT32
    new = {
        gguf.Keys.General.NAME: MetadataDetails(string, args.name),
        f"{arch}.decision.type": MetadataDetails(string, "openjev"),
        f"{arch}.decision.temperature.choice": MetadataDetails(f32, temperature),
        "tokenizer.chat_template.systemone": MetadataDetails(string, systemone_template(args.format)),
        "tokenizer.chat_templates": MetadataDetails(gguf.GGUFValueType.ARRAY, ["systemone"], sub_type=string),
    }
    writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, new, remove_metadata=[])
    raw.unlink()
    print(f"-> {out} ({out.stat().st_size / 1e9:.2f} GB, {arch}, temperature {temperature})")


if __name__ == "__main__":
    main()
