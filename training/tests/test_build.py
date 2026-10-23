import json
import sys
import unittest
from collections import defaultdict

from helpers import CleanEnv, tmpdir, write_jsonl

import build


def session(key, project, n_prompts, start, source="ds/a", clear_first=False):
    rows, prompts = [], []
    for i in range(n_prompts):
        rows.append({"role": "user", "text": f"{key} prompt {i} about the parser", "toolUses": []})
        prompts.append({"row": len(rows) - 1, "preceded_by": "clear" if clear_first and i == 0 else None,
                        "interrupted": False, "replied": True, "point": True, "hindsight": {}})
        rows.append({"role": "assistant", "text": f"{key} reply {i}", "toolUses": []})
    return {"key": key, "source": source, "file": key + ".jsonl", "project": project, "session": key,
            "start": f"2026-01-01T{start:02d}:00:00Z", "end": f"2026-01-01T{start:02d}:30:00Z",
            "rows": rows, "prompts": prompts}


def run_build(sessions):
    with tmpdir() as tmp, CleanEnv():
        inp, out = f"{tmp}/clean.jsonl", f"{tmp}/points.jsonl"
        write_jsonl(inp, sessions)
        argv = sys.argv
        sys.argv = ["build", "--input", inp, "--out", out]
        try:
            build.main()
        finally:
            sys.argv = argv
        return [json.loads(line) for line in open(out)]


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = [session(f"s{i:03d}", f"/proj/{i % 7}", 8 + i % 5, i % 20) for i in range(160)]
        cls.points = run_build(cls.sessions)

    def test_split_by_session(self):
        splits = defaultdict(set)
        for p in self.points:
            splits[p["session_key"]].add(p["split"])
            splits[p["context_key"]].add(p["split"])
        self.assertTrue(all(len(s) == 1 for s in splits.values()))
        self.assertEqual({p["split"] for p in self.points}, {"train", "dev", "test"})

    def test_points_need_context_and_respect_caps(self):
        per = defaultdict(int)
        for p in self.points:
            per[(p["session_key"], p["split"])] += p["synthetic"] is None
            if p["synthetic"] is None and p["linked"] is None:
                self.assertGreaterEqual(p["t"], build.MIN_CONTEXT)
        for (key, split), n in per.items():
            self.assertLessEqual(n, build.CAPS[split])

    def test_splices_join_two_sessions_of_the_same_repo_in_train_only(self):
        by_key = {s["key"]: s for s in self.sessions}
        splices = [p for p in self.points if p["synthetic"] == "splice"]
        self.assertTrue(splices)
        for p in splices:
            self.assertEqual(p["split"], "train")
            self.assertNotEqual(p["context_key"], p["session_key"])
            self.assertEqual(by_key[p["context_key"]]["project"], by_key[p["session_key"]]["project"])
            self.assertTrue(p["same_project"])
            self.assertEqual(p["t"], len(by_key[p["context_key"]]["rows"]))
            self.assertEqual(p["weak"], "new_task")
        held = [p for p in self.points if p["split"] != "train"]
        self.assertFalse(any(p["synthetic"] for p in held))

    def test_a_prompt_after_clear_is_linked_to_the_previous_session(self):
        a = session("before", "/proj/x", 4, 9)
        b = session("after", "/proj/x", 3, 10, clear_first=True)
        points = run_build([a, b])
        linked = [p for p in points if p["linked"] == "clear"]
        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["context_key"], "before")
        self.assertEqual(linked[0]["t"], 8)
        self.assertEqual(linked[0]["weak"], "new_task")
        self.assertEqual(len({p["split"] for p in points}), 1)

    def test_short_acknowledgements_are_weak_continuations(self):
        self.assertEqual(build.weak_label("yes, do it", None, None), None)
        self.assertEqual(build.weak_label("ok", None, None), "continuation")
        self.assertEqual(build.weak_label("Thanks!", None, None), "continuation")
        self.assertEqual(build.weak_label("fix the parser", None, "clear"), "new_task")


if __name__ == "__main__":
    unittest.main()
