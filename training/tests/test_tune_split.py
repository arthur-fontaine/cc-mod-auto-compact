import unittest

import helpers  # noqa: F401

import tune_split


def point(i, session, context=None):
    return {"id": str(i), "session_key": session, "context_key": context or session}


class SampleTest(unittest.TestCase):
    def test_whole_sessions_until_the_target_with_their_splices(self):
        points = [point(i, f"s{i % 10}") for i in range(50)] + [point(100 + i, "other", f"s{i}") for i in range(10)]
        ids = tune_split.sample(points, target=20, seed=1)
        real = [p for p in points if p["id"] in ids and p["context_key"] == p["session_key"]]
        sessions = {p["session_key"] for p in real}
        self.assertEqual(len(real), 20)
        self.assertEqual(len(sessions), 4)
        splices = {p["context_key"] for p in points if p["id"] in ids and p["session_key"] == "other"}
        self.assertEqual(splices, sessions)
        self.assertEqual(ids, tune_split.sample(points, target=20, seed=1))


if __name__ == "__main__":
    unittest.main()
