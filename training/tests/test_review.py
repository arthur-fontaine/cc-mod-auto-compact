import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

from helpers import ROOT, tmpdir, write_jsonl

sys.path.insert(0, str(ROOT / "review"))
import export  # noqa: E402
import server  # noqa: E402
from task import CATEGORIES, LETTER_OF, SYSTEM  # noqa: E402


def probs(choice, p):
    rest = (1 - p) / 4
    return {c: (p if c == choice else rest) for c in CATEGORIES}


def seed(data):
    points, states, labels = [], [], []
    for i in range(12):
        split = "test" if i < 6 else "train"
        pid = f"p{i}"
        points.append({"id": pid, "split": split, "session_key": "s1", "context_key": "s1", "t": 2,
                       "message": "next", "synthetic": None, "linked": None, "preceded_by": None})
        states.append({"id": pid, "split": split, "state": {"conversation": "0+1|user: hi", "latest_user_message": "next"}})
        labels.append({"id": pid, "split": split, "choice": "related", "probabilities": probs("related", 0.3 + i * 0.05)})
    (data / "labels").mkdir(parents=True)
    write_jsonl(data / "points.jsonl", points)
    write_jsonl(data / "states.jsonl", states)
    write_jsonl(data / "labels" / "mammoth.jsonl", labels)
    write_jsonl(data / "clean.jsonl", [{"key": "s1", "rows": [
        {"role": "user", "text": "hi", "toolUses": []}, {"role": "assistant", "text": "hello", "toolUses": []},
        {"role": "user", "text": "next", "toolUses": []}]}])


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir()
        self.data = Path(self.tmp.name)
        seed(self.data)
        self.store = server.Store(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_queue_puts_low_confidence_first_and_hides_the_teacher_when_blind(self):
        blind = self.store.queue("blind")
        self.assertEqual([q["id"] for q in blind[:3]], ["p0", "p1", "p2"])
        self.assertTrue(all(q["confidence"] is None for q in blind))
        self.assertTrue(all(self.store.points[q["id"]]["split"] == "test" for q in blind))
        self.assertNotIn("teacher", self.store.point("p0", "blind"))
        self.assertEqual(self.store.point("p7", "assisted")["teacher"]["choice"], "related")

    def test_high_confidence_points_are_sampled(self):
        queue = self.store.queue("assisted")
        conf = [q["confidence"] for q in queue]
        self.assertTrue(all(c < server.HIGH_CONFIDENCE for c in conf[:-1]) or len(conf) <= 6)

    def test_an_answer_appends_one_line_with_mode_time_and_teacher(self):
        human = self.data / "labels" / "human.jsonl"
        out = self.store.answer({"id": "p0", "mode": "blind", "label": "independent", "note": "new repo"})
        self.assertEqual(out["teacher"]["choice"], "related")
        self.store.answer({"id": "p1", "mode": "blind", "label": "skip"})
        lines = human.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual((first["mode"], first["label"], first["teacher"]["choice"]), ("blind", "independent", "related"))
        self.assertIn("timestamp", first)
        self.assertNotIn("p0", [q["id"] for q in self.store.queue("blind")])
        with self.assertRaises(ValueError):
            self.store.answer({"id": "p2", "mode": "blind", "label": "nonsense"})

    def test_http_server_answers_on_loopback(self):
        httpd = server.serve(self.store, 0)
        port = httpd.server_address[1]
        self.assertEqual(httpd.server_address[0], "127.0.0.1")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(base + "/?mode=blind") as r:
                self.assertEqual(r.status, 200)
                page = r.read().decode()
            self.assertNotIn("http://", page.replace(base, ""))
            self.assertNotIn("https://", page)
            with urllib.request.urlopen(base + "/api/point?mode=blind&id=p0") as r:
                self.assertNotIn("probabilities", r.read().decode())
            req = urllib.request.Request(base + "/api/label", data=json.dumps(
                {"id": "p3", "mode": "blind", "label": "related"}).encode(), method="POST")
            with urllib.request.urlopen(req) as r:
                self.assertEqual(json.loads(r.read())["teacher"]["choice"], "related")
        finally:
            httpd.shutdown()


class ExportTest(unittest.TestCase):
    def test_tev1_records_from_human_labels(self):
        points = {"a": {"split": "test", "session_key": "s"}, "b": {"split": "train", "session_key": "t"}}
        states = {k: {"conversation": "0+1|user: hi", "latest_user_message": "go"} for k in points}
        human = [{"id": "a", "label": "related", "mode": "blind"}, {"id": "a", "label": "independent", "mode": "blind"},
                 {"id": "b", "label": "skip", "mode": "assisted"}]
        recs = export.records(human, points, states)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r["split"], r["completion"]), ("test", LETTER_OF["independent"]))
        self.assertEqual(r["prompt"][0], {"role": "system", "content": SYSTEM})
        task = json.loads(r["prompt"][1]["content"])
        self.assertEqual([o["key"] for o in task["options"]], CATEGORIES)
        self.assertEqual(task["state"]["latest_user_message"], "go")


if __name__ == "__main__":
    unittest.main()
