"""Label decision points with the teacher, Baseten Mammoth, through the System One API.

    uv run python pipeline/teacher.py --dry-run         # estimate only
    uv run python pipeline/teacher.py                   # data/states.jsonl -> data/labels/mammoth.jsonl

Every answer is cached by hash(model slug, request) in data/teacher-cache/, so a rerun
never asks a cached point again. Per call it stores the whole probability distribution,
the latency, the input tokens and the served version (`x-baseten-model-version-id`);
if the version changes during a run, labeling stops and says so.

Limits (the rate limit is unknown, the key a free trial): concurrency starts at 2 and
rises slowly; an HTTP 429 halves it and backs off exponentially with jitter, and is
logged with its headers (redacted) to data/teacher-log.jsonl. A credit or quota failure
(HTTP 402; a 403 or 429 whose body mentions credit, billing, quota, trial or balance; or 5
consecutive 429s after backoff) stops every call at once, keeps the cache, notifies,
and prints a block starting with BASETEN STOP. scripts/teacher_status.py reads the
same files without calling the API.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, guard_spend, load_env, redact, redact_headers, require  # noqa: E402
from task import CATEGORIES, QUESTION_ID  # noqa: E402

DEFAULT_ENDPOINT = "https://inference.baseten.co/v1/systemone"
CACHE = DATA / "teacher-cache"
LOG = DATA / "teacher-log.jsonl"
STATE = DATA / "teacher-state.json"
OUT = DATA / "labels" / "mammoth.jsonl"
# The free week on Baseten's Model APIs ends Oct 27, 2026, 11:59 PM PT.
FREE_UNTIL = datetime(2026, 10, 28, 6, 59, 59, tzinfo=timezone.utc)
STOP_WORDS = ("credit", "billing", "quota", "trial", "balance")
MAX_CONSECUTIVE_429 = 5
MAX_CONCURRENCY = 8
RESUME = "cd training && uv run python pipeline/teacher.py"


class TeacherStop(Exception):
    def __init__(self, status, body, reason):
        super().__init__(reason)
        self.status, self.body, self.reason = status, body, reason


class VersionChanged(Exception):
    pass


def classify(status, body, consecutive_429=0):
    """What to do after a failed call: 'stop', 'backoff', 'retry' or 'fail'."""
    text = (body or "").lower()
    if status == 402:
        return "stop"
    if status in (403, 429) and any(w in text for w in STOP_WORDS):
        return "stop"
    if status == 429:
        return "stop" if consecutive_429 >= MAX_CONSECUTIVE_429 else "backoff"
    if status is None or status >= 500:
        return "retry"
    return "fail"


def cache_key(model, body):
    return hashlib.sha256((model + "\n" + json.dumps(body, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


class Cache:
    def __init__(self, directory=CACHE):
        self.path = directory / "answers.jsonl"
        self.entries = {}
        if self.path.exists():
            for line in self.path.open():
                e = json.loads(line)
                self.entries[e["key"]] = e
        self.lock = threading.Lock()

    def get(self, key):
        return self.entries.get(key)

    def put(self, entry):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.entries[entry["key"]] = entry


def post(endpoint, key, body, timeout=120):
    """(status, headers, text) for one call; status None when no response arrived."""
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={
        "Authorization": "Api-Key " + key, "Content-Type": "application/json",
        "User-Agent": "spisev-teacher/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read().decode("utf-8", "replace")
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"


def log(event):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n")


def parse_answer(text):
    data = json.loads(text)
    answer = data["answers"][QUESTION_ID]
    probs = {c: float(answer["probabilities"].get(c, 0.0)) for c in CATEGORIES}
    return {"choice": answer["choice"], "probabilities": probs, "confidence": answer.get("confidence"),
            "model": data.get("model"), "input_tokens": (data.get("usage") or {}).get("input_tokens")}


class Labeler:
    def __init__(self, endpoint, key, model, cache, post_fn=post, sleep=time.sleep, start_concurrency=2):
        self.endpoint, self.key, self.model, self.cache = endpoint, key, model, cache
        self.post, self.sleep = post_fn, sleep
        self.concurrency = start_concurrency
        self.in_flight = 0
        self.cond = threading.Condition()
        self.stopped = None
        self.version = json.loads(STATE.read_text()).get("version") if STATE.exists() else None
        self.consecutive_429 = 0
        self.successes = 0
        self.calls = 0
        self.count_429 = 0

    def _acquire(self):
        with self.cond:
            while self.in_flight >= self.concurrency and not self.stopped:
                self.cond.wait(0.5)
            if self.stopped:
                raise self.stopped
            self.in_flight += 1

    def _release(self):
        with self.cond:
            self.in_flight -= 1
            self.cond.notify_all()

    def _stop(self, error):
        with self.cond:
            if not self.stopped:
                self.stopped = error
            self.cond.notify_all()

    def label(self, body):
        """The teacher's answer for one request, from the cache when it has it. An answer cached
        under another served version than the one accepted is asked again, once."""
        key = cache_key(self.model, body)
        cached = self.cache.get(key)
        if cached and (self.version is None or cached.get("version") in (None, self.version)):
            return cached, False
        attempt = 0
        while True:
            if self.stopped:
                raise self.stopped
            self._acquire()
            try:
                started = time.perf_counter()
                self.calls += 1
                status, headers, text = self.post(self.endpoint, self.key, body)
                latency = (time.perf_counter() - started) * 1000
            finally:
                self._release()
            if status == 200:
                self.consecutive_429 = 0
                version = headers.get("x-baseten-model-version-id") or headers.get("X-Baseten-Model-Version-Id")
                answer = parse_answer(text)
                with self.cond:
                    if self.version is None and version:
                        self.version = version
                        STATE.parent.mkdir(parents=True, exist_ok=True)
                        STATE.write_text(json.dumps({"version": version, "model": self.model}))
                    elif version and version != self.version:
                        self._stop(VersionChanged(f"served version changed from {self.version} to {version}"))
                        raise self.stopped
                    self.successes += 1
                    if self.successes % 50 == 0 and self.concurrency < MAX_CONCURRENCY:
                        self.concurrency += 1
                entry = {"key": key, "model": self.model, "version": version, "latency_ms": round(latency, 1),
                         **answer, "ratelimit": {k: v for k, v in headers.items() if "ratelimit" in k.lower()}}
                self.cache.put(entry)
                return entry, True
            body_text = redact(text, [self.key])[:2000]
            with self.cond:
                if status == 429:
                    self.count_429 += 1
                    self.consecutive_429 += 1
                consecutive = self.consecutive_429
            decision = classify(status, body_text, consecutive)
            log({"status": status, "decision": decision, "body": body_text[:500],
                 "headers": redact_headers(headers) if status == 429 else None})
            if decision == "stop":
                self._stop(TeacherStop(status, body_text, "credit or quota failure"
                                       if status != 429 or any(w in body_text.lower() for w in STOP_WORDS)
                                       else f"{consecutive} consecutive 429s after backoff"))
                raise self.stopped
            if decision == "backoff":
                with self.cond:
                    self.concurrency = max(1, self.concurrency // 2)
                self.sleep(min(120, 2 ** consecutive) + random.uniform(0, 1))
                continue
            attempt += 1
            if decision == "fail" or attempt >= 4:
                raise RuntimeError(f"HTTP {status}: {body_text[:300]}")
            self.sleep(2 ** attempt + random.uniform(0, 1))


def notify_stop():
    subprocess.run(["osascript", "-e", 'display notification "Baseten: credit or quota problem, labeling stopped" '
                                       'with title "spisev" sound name "Glass"'], check=False)


def stop_block(error, labeled, left):
    return "\n".join([
        "BASETEN STOP",
        f"  HTTP status: {error.status}",
        f"  reason: {error.reason}",
        f"  body: {error.body}",
        f"  points labeled so far: {labeled}",
        f"  points left: {left}",
        f"  resume: {RESUME}",
    ])


def billed_price(now=None):
    now = now or datetime.now(timezone.utc)
    if now <= FREE_UNTIL:
        return 0.0
    price = os.environ.get("DECISION_PRICE_PER_MTOK")
    if not price:
        raise SystemExit("Mammoth's free week is over and DECISION_PRICE_PER_MTOK is not set: refusing to spend "
                         "at an unknown price.")
    return float(price)


def settings():
    load_env()
    key, model = require("BASETEN_API_KEY", "DECISION_MODEL")
    return os.environ.get("DECISION_ENDPOINT") or DEFAULT_ENDPOINT, key, model


def main():
    from task import request, state_size

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", type=Path, default=DATA / "states.jsonl")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-version", help="Label with this newly served version (answers of other "
                                                 "versions are asked again); the previous one is recorded")
    args = parser.parse_args()
    endpoint, key, model = settings()
    if args.accept_version:
        old = json.loads(STATE.read_text()) if STATE.exists() else {}
        STATE.write_text(json.dumps({"version": args.accept_version, "model": model,
                                     "previous": old.get("previous", []) + ([old["version"]] if old.get("version") else []),
                                     "accepted_at": time.time()}))
        print(f"[mammoth] accepted served version {args.accept_version} (was {old.get('version')})", flush=True)

    rows = [json.loads(line) for line in args.states.open()]
    rows = [r for r in rows if r["split"] in args.splits.split(",")]
    if args.limit_sessions:
        keep = list(dict.fromkeys(r["session_key"] for r in rows))[: args.limit_sessions]
        rows = [r for r in rows if r["session_key"] in set(keep)]
    cache = Cache()
    bodies = [request(r["state"], model) for r in rows]
    current = json.loads(STATE.read_text()).get("version") if STATE.exists() else None
    todo = [b for b in bodies if not cache.get(cache_key(model, b))
            or (current and cache.get(cache_key(model, b)).get("version") not in (None, current))]
    tokens = sum(state_size(b["state"])["est_tokens"] for b in todo)
    if not guard_spend("mammoth", calls=len(todo), billed_usd=tokens * billed_price() / 1e6, list_usd=None,
                       dry_run=args.dry_run):
        return
    print(f"[mammoth] {len(rows)} points, {len(rows) - len(todo)} cached, {len(todo)} to ask "
          f"(~{tokens / 1e6:.2f}M input tokens) at {endpoint}", flush=True)

    labeler = Labeler(endpoint, key, model, cache)
    results, failures, new_tokens = {}, 0, 0
    lock = threading.Lock()

    def work(i):
        nonlocal failures, new_tokens
        try:
            entry, new = labeler.label(bodies[i])
            with lock:
                results[i] = entry
                new_tokens += (entry.get("input_tokens") or 0) if new else 0
        except (TeacherStop, VersionChanged):
            pass
        except Exception as e:
            with lock:
                failures += 1
            print(f"point {rows[i]['id']}: {e}", file=sys.stderr, flush=True)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
        for n, _ in enumerate(cf.as_completed([ex.submit(work, i) for i in range(len(rows))]), 1):
            if n % 200 == 0:
                print(f"[mammoth] {n}/{len(rows)} · concurrency {labeler.concurrency} · 429s {labeler.count_429}",
                      flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, r in enumerate(rows):
            if i in results:
                e = results[i]
                f.write(json.dumps({"id": r["id"], "split": r["split"], "choice": e["choice"],
                                    "probabilities": e["probabilities"], "confidence": e.get("confidence"),
                                    "version": e.get("version"), "latency_ms": e.get("latency_ms"),
                                    "input_tokens": e.get("input_tokens")}) + "\n")
    print(f"[mammoth] spend: {labeler.calls} new calls, {new_tokens} new input tokens, billed "
          f"${new_tokens * billed_price() / 1e6:.2f}; {len(results)} labeled, {failures} failed -> {args.out}")
    if isinstance(labeler.stopped, TeacherStop):
        notify_stop()
        print(stop_block(labeler.stopped, len(results), len(rows) - len(results)))
        sys.exit(3)
    if isinstance(labeler.stopped, VersionChanged):
        print(f"MAMMOTH VERSION CHANGED: {labeler.stopped}. Labels so far are kept; decide whether to relabel, "
              f"then rerun: {RESUME}")
        sys.exit(4)


if __name__ == "__main__":
    main()
