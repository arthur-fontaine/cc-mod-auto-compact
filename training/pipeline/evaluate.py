"""Evaluate an alpha candidate, or Mammoth, on the held-out test-candidate sessions.

    uv run python pipeline/evaluate.py --model runs/qwen3-1.7b/fused-8bit --name qwen3-1.7b-mlx
    uv run python pipeline/evaluate.py --endpoint http://127.0.0.1:8790/v1/systemone --name qwen3-1.7b
    uv run python pipeline/evaluate.py --mammoth --name mammoth          # its latency, one request at a time
    uv run python pipeline/evaluate.py --cloud decision-models/baseten-elephant-9b-latest --name elephant
    uv run python pipeline/alpha_report.py                               # the section 11 table

Mammoth is the reference: top-1 agreement with its pick, KL to its distribution,
calibration error, and the action agreement through policy.py. The precision of
`independent` against the weak labels is reported for every row, Mammoth's included.
Latency is end to end, one request at a time. Results go to results/NAME.json.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score  # noqa: E402
from task import CATEGORIES, QUESTION_ID, request  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def ask_endpoint(endpoint, state, model="spisev-alpha"):
    body = json.dumps(request(state, model)).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        answer = json.load(r)["answers"][QUESTION_ID]
    return {c: float(answer["probabilities"].get(c, 0.0)) for c in CATEGORIES}, (time.perf_counter() - started) * 1000


def percentiles(ms):
    ms = sorted(ms)
    return {"p50": ms[len(ms) // 2], "p95": ms[min(len(ms) - 1, int(len(ms) * 0.95))], "max": ms[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="An MLX model directory")
    parser.add_argument("--endpoint", help="A System One endpoint (llama-server)")
    parser.add_argument("--mammoth", action="store_true", help="Re-ask Mammoth one request at a time for latency")
    parser.add_argument("--cloud", metavar="MODEL", help="Another model on the teacher's System One endpoint")
    parser.add_argument("--api-model", default="spisev-alpha", help="With --endpoint: the request's model")
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from config import load_env
    load_env()

    rows = [json.loads(line) for line in (ROOT / "data" / "split" / f"{args.split}.jsonl").open()][: args.limit]
    preds = []
    if args.model:
        from infer import Classifier
        clf = Classifier(args.model)
        clf.predict(rows[0]["state"])  # warm up
        for e in rows:
            p = clf.predict(e["state"])
            preds.append((p["probabilities"], p["ms"]))
    elif args.endpoint:
        ask_endpoint(args.endpoint, rows[0]["state"], args.api_model)  # warm up
        for e in rows:
            started = time.perf_counter()
            try:
                preds.append(ask_endpoint(args.endpoint, e["state"], args.api_model))
            except Exception as err:  # the mod keeps the context on an error; the benchmark scores it so
                print(f"{e['id']}: {str(err)[:120]}", file=sys.stderr, flush=True)
                preds.append((None, (time.perf_counter() - started) * 1000))
    elif args.mammoth or args.cloud:
        import tempfile
        import teacher
        endpoint, key, model = teacher.settings()
        model = args.cloud or model
        with tempfile.TemporaryDirectory() as tmp:
            # Another model is served under its own versions: keep them out of the teacher's state.
            state = Path(tmp) / "state.json" if args.cloud else teacher.STATE
            labeler = teacher.Labeler(endpoint, key, model, teacher.Cache(Path(tmp)), start_concurrency=1, state=state)
            for e in rows:
                entry, _ = labeler.label(request(e["state"], model))
                preds.append((entry["probabilities"], entry["latency_ms"]))
    else:
        parser.error("give --model, --endpoint, --mammoth or --cloud")

    answered = [(p, e) for (p, _), e in zip(preds, rows) if p is not None]
    scored = score([{"probs": p, "reference": dict(zip(CATEGORIES, e["targets"])), "weak": e.get("weak")}
                    for p, e in answered])
    scored["errors"] = len(rows) - len(answered)
    agrees = iter(scored["correct"])
    scored["correct"] = [next(agrees) if p is not None else False for p, _ in preds]
    result = {"name": args.name, "split": args.split, **{k: v for k, v in scored.items() if k != "correct"},
              "latency_ms": percentiles([ms for _, ms in preds])}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=1))
    with (out / f"{args.name}.predictions.jsonl").open("w") as f:
        for (p, ms), e, ok in zip(preds, rows, scored["correct"]):
            f.write(json.dumps({"id": e["id"], "probabilities": p, "ms": ms, "agrees": ok}) + "\n")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
