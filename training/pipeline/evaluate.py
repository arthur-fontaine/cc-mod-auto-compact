"""Evaluate an alpha candidate, or Mammoth, on the held-out test-candidate sessions.

    uv run python pipeline/evaluate.py --model runs/qwen3-1.7b/fused-8bit --name qwen3-1.7b-mlx
    uv run python pipeline/evaluate.py --endpoint http://127.0.0.1:8790/v1/systemone --name qwen3-1.7b
    uv run python pipeline/evaluate.py --endpoint ... --warm --name qwen3-1.7b-warm   # latency after send
    uv run python pipeline/evaluate.py --mammoth --name mammoth          # its latency, one request at a time
    uv run python pipeline/evaluate.py --mammoth --states data/states-v2-160.jsonl --split dev --name mammoth-v2-160-dev
    uv run python pipeline/evaluate.py --cloud decision-models/baseten-elephant-9b-latest --name elephant
    uv run python pipeline/evaluate.py --claude claude-haiku-5-5 --split dev --name haiku-v1-dev   # claude_teacher.py
    uv run python pipeline/alpha_report.py                               # the section 11 table

Mammoth is the reference: top-1 agreement with its pick, KL to its distribution,
calibration error, and the action agreement through policy.py. The precision of
`independent` against the weak labels is reported for every row, Mammoth's included.
Latency is end to end, one request at a time. With --warm, each state is first sent without
its new message (untimed), as the mod prefills it while the user types, and only the full
request is timed. Results go to results/NAME.json.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score  # noqa: E402
from task import CATEGORIES, QUESTION_ID, request, request_v2, state_size  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def request_for(state, model):
    """The System One request of a v1 or a v2 state."""
    return (request_v2 if "memory" in state else request)(state, model)


def ask_endpoint(endpoint, state, model="spisev-alpha"):
    body = json.dumps(request_for(state, model)).encode()
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
    parser.add_argument("--claude", metavar="MODEL", help="A Claude model through claude -p (claude_teacher.py)")
    parser.add_argument("--api-model", default="spisev-alpha", help="With --endpoint: the request's model")
    parser.add_argument("--message-tokens", type=int, help="Clip each new message to this many (estimated) tokens")
    parser.add_argument("--warm", action="store_true", help="With --endpoint: prefill each state without its message first")
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--states", type=Path, help="Ask with these states (build_tree.py output) for the split's points")
    parser.add_argument("--keep-cache", action="store_true",
                        help="With --mammoth: use the teacher's answer cache (no fresh latency, no repeat calls)")
    parser.add_argument("--eval-cache", action="store_true",
                        help="With --mammoth: a cache and a version pin of their own (data/teacher-cache-eval), so an "
                             "evaluation can follow a newly served version without touching the training labels'")
    parser.add_argument("--dry-run", action="store_true", help="With --mammoth or --cloud: print the estimate only")
    args = parser.parse_args()
    from config import load_env
    load_env()

    rows = [json.loads(line) for line in (ROOT / "data" / "split" / f"{args.split}.jsonl").open()][: args.limit]
    if args.states:
        states = {s["id"]: s["state"] for s in map(json.loads, args.states.open())}
        missing = [e["id"] for e in rows if e["id"] not in states]
        if missing:
            raise SystemExit(f"{len(missing)} points of {args.split} have no state in {args.states}")
        rows = [{**e, "state": states[e["id"]]} for e in rows]
    if args.message_tokens:
        from text import clip
        from task import MAX_MESSAGE_BYTES
        for e in rows:
            message = e["state"]["latest_user_message"]
            e["state"] = {**e["state"], "latest_user_message": clip(message, MAX_MESSAGE_BYTES, args.message_tokens)}
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
            if args.warm:
                ask_endpoint(args.endpoint, {**e["state"], "latest_user_message": ""}, args.api_model)
            started = time.perf_counter()
            try:
                preds.append(ask_endpoint(args.endpoint, e["state"], args.api_model))
            except Exception as err:  # the mod keeps the context on an error; the benchmark scores it so
                print(f"{e['id']}: {str(err)[:120]}", file=sys.stderr, flush=True)
                preds.append((None, (time.perf_counter() - started) * 1000))
    elif args.mammoth or args.cloud:
        import tempfile
        import teacher
        from config import guard_spend
        endpoint, key, model = teacher.settings()
        model = args.cloud or model
        eval_dir = ROOT / "data" / "teacher-cache-eval"
        known = teacher.Cache(eval_dir) if args.eval_cache else teacher.Cache() if args.keep_cache and not args.cloud else None
        todo = [e for e in rows if not (known and known.get(teacher.cache_key(model, request_for(e["state"], model))))]
        tokens = sum(state_size(e["state"])["est_tokens"] for e in todo)
        billed = tokens * teacher.billed_price() / 1e6 if todo and not args.cloud else 0.0
        if not guard_spend(args.cloud or "mammoth", calls=len(todo), billed_usd=billed, list_usd=None,
                           dry_run=args.dry_run):
            return
        with tempfile.TemporaryDirectory() as tmp:
            # Another model is served under its own versions: keep them out of the teacher's state.
            state = Path(tmp) / "state.json" if args.cloud else eval_dir / "state.json" if args.eval_cache else teacher.STATE
            cache = known or teacher.Cache(Path(tmp))
            labeler = teacher.Labeler(endpoint, key, model, cache, start_concurrency=1, state=state)
            for e in rows:
                entry, _ = labeler.label(request_for(e["state"], model))
                preds.append((entry["probabilities"], entry["latency_ms"]))
    elif args.claude:
        import concurrent.futures as cf
        import claude_teacher
        from config import guard_spend
        if not guard_spend(args.claude, calls=len(rows), billed_usd=0.0, list_usd=None, dry_run=args.dry_run):
            return

        def one(e):
            try:
                category, ms = claude_teacher.ask(args.claude, e["state"])
            except Exception as err:
                print(f"{e['id']}: {str(err)[:120]}", file=sys.stderr, flush=True)
                return None, 0.0
            return {c: float(c == category) for c in CATEGORIES}, ms
        with cf.ThreadPoolExecutor(4) as pool:
            for i, pred in enumerate(pool.map(one, rows), 1):
                preds.append(pred)
                if i % 25 == 0:
                    print(f"[{args.claude}] {i}/{len(rows)}", file=sys.stderr, flush=True)
    else:
        parser.error("give --model, --endpoint, --mammoth, --cloud or --claude")

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
