"""Review site for Mammoth's labels, on 127.0.0.1 only, with no external assets.

    uv run python review/server.py            # http://127.0.0.1:8377

Per point: the raw transcript around the boundary (collapsible), the state as the
teacher saw it, and Mammoth's pick and probabilities.

- Blind mode (test-candidates): the teacher's answer stays hidden until you answer.
- Assisted mode (train): it is shown.
- Queue: lowest confidence first, plus a random 20% of the high-confidence points.
- Keys: 1-5 pick a category, s skips, f marks the summary insufficient, n adds a note.

Answers append to data/labels/human.jsonl (mode, timestamp, teacher answer); nothing
is ever rewritten.
"""
import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from config import DATA  # noqa: E402
from task import CATEGORIES, CRITERIA  # noqa: E402
from text import messages  # noqa: E402

HUMAN = DATA / "labels" / "human.jsonl"
MODES = {"blind": "test", "assisted": "train"}
HIGH_CONFIDENCE = 0.8
HIGH_SAMPLE = 0.2
PAGE = (Path(__file__).resolve().parent / "page.html").read_text()


def read(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


class Store:
    def __init__(self, data=DATA, human=None):
        self.data = data
        self.human = human or data / "labels" / "human.jsonl"
        self.lock = threading.Lock()
        self.points = {p["id"]: p for p in read(data / "points.jsonl")}
        self.states = {s["id"]: s["state"] for s in read(data / "states.jsonl")}
        self.labels = {lab["id"]: lab for lab in read(data / "labels" / "mammoth.jsonl")}
        self._sessions = None

    def sessions(self):
        if self._sessions is None:
            wanted = {k for p in self.points.values() for k in (p["context_key"], p["session_key"])}
            self._sessions = {}
            path = self.data / "clean.jsonl"
            if path.exists():
                for line in path.open():
                    s = json.loads(line)
                    if s["key"] in wanted:
                        self._sessions[s["key"]] = s
        return self._sessions

    def answered(self, mode):
        return {r["id"] for r in read(self.human) if r["mode"] == mode}

    def queue(self, mode, seed=0):
        split = MODES[mode]
        ids = [i for i, p in self.points.items() if p["split"] == split and i in self.labels and i in self.states]
        done = self.answered(mode)
        ids = [i for i in ids if i not in done]
        conf = {i: max(self.labels[i]["probabilities"].values()) for i in ids}
        low = sorted((i for i in ids if conf[i] < HIGH_CONFIDENCE), key=lambda i: (conf[i], i))
        high = sorted(i for i in ids if conf[i] >= HIGH_CONFIDENCE)
        rng = random.Random(seed)
        sample = rng.sample(high, round(len(high) * HIGH_SAMPLE)) if high else []
        return [{"id": i, "confidence": None if mode == "blind" else round(conf[i], 3)} for i in low + sample]

    def transcript(self, p, around=8):
        """Messages before the boundary and the turn it starts, as the raw session holds them."""
        sessions = self.sessions()
        ctx = sessions.get(p["context_key"])
        before = messages(ctx["rows"])[: p["t"]][-around:] if ctx else []
        after = []
        own = sessions.get(p["session_key"])
        if own:
            msgs = messages(own["rows"])
            if p["context_key"] == p["session_key"]:
                after = msgs[p["t"]: p["t"] + 2]
            else:
                after = msgs[:2]
        return {"before": [{"kind": m["kind"], "text": m["text"][:4000]} for m in before],
                "after": [{"kind": m["kind"], "text": m["text"][:4000]} for m in after]}

    def point(self, pid, mode):
        p = self.points[pid]
        out = {"id": pid, "split": p["split"], "synthetic": p["synthetic"], "linked": p["linked"],
               "preceded_by": p["preceded_by"], "state": self.states.get(pid),
               "transcript": self.transcript(p), "categories": CATEGORIES, "criteria": CRITERIA}
        if mode == "assisted":
            out["teacher"] = self.teacher(pid)
        return out

    def teacher(self, pid):
        lab = self.labels.get(pid)
        return {"choice": lab["choice"], "probabilities": lab["probabilities"]} if lab else None

    def answer(self, body):
        pid, mode = body["id"], body["mode"]
        if pid not in self.points or mode not in MODES:
            raise ValueError("unknown point or mode")
        label = body.get("label")
        if label not in CATEGORIES + ["skip", "insufficient"]:
            raise ValueError(f"label must be one of {CATEGORIES + ['skip', 'insufficient']}")
        record = {"id": pid, "mode": mode, "split": self.points[pid]["split"], "label": label,
                  "note": body.get("note") or None, "timestamp": time.time(), "teacher": self.teacher(pid)}
        with self.lock:
            self.human.parent.mkdir(parents=True, exist_ok=True)
            with self.human.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"saved": True, "teacher": record["teacher"]}


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, kind="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            mode = q.get("mode", "blind")
            if mode not in MODES:
                return self.send(400, {"error": "mode must be blind or assisted"})
            if url.path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if url.path == "/api/queue":
                return self.send(200, {"mode": mode, "queue": store.queue(mode)})
            if url.path == "/api/point" and q.get("id") in store.points:
                return self.send(200, store.point(q["id"], mode))
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                return self.send(200, store.answer(body))
            except (ValueError, KeyError) as e:
                return self.send(400, {"error": str(e)})

    return Handler


def serve(store, port):
    return ThreadingHTTPServer(("127.0.0.1", port), handler(store))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8377)
    args = parser.parse_args()
    server = serve(Store(), args.port)
    print(f"review site: http://127.0.0.1:{args.port}/ (blind: ?mode=blind, assisted: ?mode=assisted)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
