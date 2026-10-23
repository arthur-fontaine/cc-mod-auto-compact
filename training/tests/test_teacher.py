import json
import unittest
from unittest import mock

from helpers import CleanEnv, tmpdir

import teacher
from task import CATEGORIES


def ok(choice="related", version="v1"):
    body = {"model": "m", "answers": {"boundary": {"type": "choice", "choice": choice, "confidence": 0.5,
                                                    "probabilities": {c: (0.6 if c == choice else 0.1) for c in CATEGORIES}}},
            "usage": {"input_tokens": 100, "output_tokens": 1}}
    return 200, {"x-baseten-model-version-id": version, "x-ratelimit-remaining-requests": "10"}, json.dumps(body)


class Script:
    """A stub teacher endpoint answering from a list; it never touches the network."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, endpoint, key, body):
        self.calls += 1
        return self.responses.pop(0) if self.responses else ok()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir()
        d = teacher.Path(self.tmp.name)
        self.patches = [mock.patch.object(teacher, "LOG", d / "log.jsonl"),
                        mock.patch.object(teacher, "STATE", d / "state.json")]
        for p in self.patches:
            p.start()
        self.cache = teacher.Cache(d / "cache")
        self.env = CleanEnv(BASETEN_API_KEY="bt-secret-key-0123456789")
        self.env.__enter__()

    def tearDown(self):
        self.env.__exit__()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def labeler(self, script):
        return teacher.Labeler("https://x", "bt-secret-key-0123456789", "slug", self.cache, post_fn=script,
                               sleep=lambda s: None)


class ClassifyTest(unittest.TestCase):
    def test_classifier(self):
        self.assertEqual(teacher.classify(429, "slow down", 1), "backoff")
        self.assertEqual(teacher.classify(429, "slow down", 5), "stop")
        self.assertEqual(teacher.classify(402, "", 0), "stop")
        for word in ("credit", "Billing", "QUOTA", "trial", "balance"):
            self.assertEqual(teacher.classify(403, f"your {word} is exhausted"), "stop", word)
            self.assertEqual(teacher.classify(429, f"{word} exceeded"), "stop", word)
        self.assertEqual(teacher.classify(403, "forbidden"), "fail")
        self.assertEqual(teacher.classify(503, "down"), "retry")
        self.assertEqual(teacher.classify(None, "timeout"), "retry")
        self.assertEqual(teacher.classify(400, "bad request"), "fail")


class LabelerTest(Base):
    def test_429_backs_off_halves_concurrency_and_is_logged_redacted(self):
        script = Script([(429, {"retry-after": "1", "Authorization": "Api-Key bt-secret-key-0123456789"}, "rate"),
                         ok()])
        lab = self.labeler(script)
        lab.concurrency = 4
        entry, new = lab.label({"state": {}, "n": 1})
        self.assertTrue(new)
        self.assertEqual(entry["choice"], "related")
        self.assertEqual(lab.concurrency, 2)
        log = teacher.LOG.read_text()
        self.assertIn('"status": 429', log)
        self.assertNotIn("bt-secret-key-0123456789", log)

    def test_402_stops_with_no_retry_and_no_later_call(self):
        script = Script([(402, {}, '{"error": "payment required for key bt-secret-key-0123456789"}')])
        lab = self.labeler(script)
        with self.assertRaises(teacher.TeacherStop) as ctx:
            lab.label({"n": 1})
        self.assertEqual(script.calls, 1)
        self.assertNotIn("bt-secret-key-0123456789", ctx.exception.body)
        with self.assertRaises(teacher.TeacherStop):
            lab.label({"n": 2})
        self.assertEqual(script.calls, 1)

    def test_quota_body_on_429_stops(self):
        script = Script([(429, {}, '{"error": "free trial quota exhausted"}')])
        with self.assertRaises(teacher.TeacherStop):
            self.labeler(script).label({"n": 1})
        self.assertEqual(script.calls, 1)

    def test_five_consecutive_429s_stop(self):
        script = Script([(429, {}, "slow down")] * 10)
        with self.assertRaises(teacher.TeacherStop) as ctx:
            self.labeler(script).label({"n": 1})
        self.assertEqual(script.calls, teacher.MAX_CONSECUTIVE_429)
        self.assertIn("consecutive", ctx.exception.reason)

    def test_cached_points_are_never_asked_again(self):
        script = Script([ok("independent")])
        lab = self.labeler(script)
        lab.label({"n": 1})
        entry, new = self.labeler(script).label({"n": 1})
        self.assertFalse(new)
        self.assertEqual(entry["choice"], "independent")
        self.assertEqual(script.calls, 1)
        self.assertEqual(len(teacher.Cache(teacher.Path(self.tmp.name) / "cache").entries), 1)

    def test_a_changed_served_version_stops(self):
        script = Script([ok(version="v1"), ok(version="v2")])
        lab = self.labeler(script)
        lab.label({"n": 1})
        with self.assertRaises(teacher.VersionChanged):
            lab.label({"n": 2})

    def test_an_answer_of_another_served_version_is_asked_again_once(self):
        script = Script([ok("related", version="v1"), ok("independent", version="v2")])
        self.labeler(script).label({"n": 1})
        teacher.STATE.write_text('{"version": "v2"}')
        lab = self.labeler(script)
        entry, new = lab.label({"n": 1})
        self.assertTrue(new)
        self.assertEqual((entry["choice"], entry["version"]), ("independent", "v2"))
        self.assertFalse(self.labeler(script).label({"n": 1})[1])
        self.assertEqual(script.calls, 2)

    def test_cache_stores_distribution_latency_tokens_and_no_headers_with_keys(self):
        entry, _ = self.labeler(Script([ok()])).label({"n": 1})
        self.assertEqual(set(entry["probabilities"]), set(CATEGORIES))
        self.assertIn("latency_ms", entry)
        self.assertEqual(entry["input_tokens"], 100)
        self.assertEqual(entry["version"], "v1")
        self.assertNotIn("bt-secret", json.dumps(entry))

    def test_stop_block(self):
        err = teacher.TeacherStop(402, "payment required", "credit or quota failure")
        block = teacher.stop_block(err, 10, 5)
        self.assertTrue(block.startswith("BASETEN STOP"))
        for part in ("402", "payment required", "10", "5", "pipeline/teacher.py"):
            self.assertIn(part, block)

    def test_spend_after_the_free_week_needs_a_price(self):
        from datetime import datetime, timezone
        self.assertEqual(teacher.billed_price(datetime(2026, 10, 23, tzinfo=timezone.utc)), 0.0)
        with self.assertRaises(SystemExit):
            teacher.billed_price(datetime(2026, 11, 1, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
