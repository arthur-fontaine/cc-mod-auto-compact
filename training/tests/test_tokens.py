import unittest

import helpers  # noqa: F401

import tokens


def chars(text):
    return [ord(c) for c in text]


class TokensTest(unittest.TestCase):
    def test_parts_add_up(self):
        state = {"conversation": "0+1|user: fix the login\n1+1|task: login fix [started]", "latest_user_message": "ok"}
        out, ids = tokens.breakdown(state, chars)
        self.assertEqual(out["total"], len(ids))
        self.assertEqual(out["fixed"] + out["conversation"] + out["message"], out["total"])
        self.assertEqual(out["message"], 2)
        self.assertEqual(out["lines"], 2)
        self.assertIsNone(out["reused"])
        # Only the message and what follows it are left once the rest is prefilled.
        self.assertLess(out["warm_left"], out["conversation"])

    def test_reused_prefix_from_previous_state(self):
        a = {"conversation": "0+1|user: a", "latest_user_message": "b"}
        b = {"conversation": "0+1|user: a\n1+1|agent: done", "latest_user_message": "c"}
        _, ids = tokens.breakdown(a, chars)
        out, _ = tokens.breakdown(b, chars, ids)
        self.assertEqual(out["reused"], tokens.common_prefix(ids, chars(tokens.render(b))))
        text = tokens.render(a)
        self.assertGreater(out["reused"], text.index('"conversation": "') + len("0+1|user: a"))

    def test_v2_parts_add_up(self):
        state = {"memory": "0+1|user: fix the login", "last_agent_message": "Fixed. Commit?", "latest_user_message": "yes"}
        out, ids = tokens.breakdown(state, chars)
        self.assertEqual(out["fixed"] + out["conversation"] + out["agent"] + out["message"], out["total"])
        self.assertEqual((out["agent"], out["message"], out["lines"]), (14, 3, 1))

    def test_line_kind(self):
        self.assertEqual(tokens.line_kind("4+4|task: x [done]"), "merged")
        self.assertEqual(tokens.line_kind("7+1|task: x [done]"), "task")
        self.assertEqual(tokens.line_kind("8+1|user: hello"), "user")
        self.assertEqual(tokens.line_kind("9+1|no label here"), "other")

    def test_bins(self):
        self.assertEqual(tokens.bin_of(0), "0-500")
        self.assertEqual(tokens.bin_of(999), "500-1000")
        self.assertEqual(tokens.bin_of(4000), "4000+")


if __name__ == "__main__":
    unittest.main()
