import asyncio
import unittest

import helpers  # noqa: F401
import build_tree
import tree
from text import utf8_len


class ViewTest(unittest.TestCase):
    def test_small_chats_are_all_leaves(self):
        self.assertEqual(tree.view(5), [(0, i) for i in range(5)])

    def test_view_covers_every_message_once_and_keeps_recent_leaves(self):
        for t in (0, 1, 33, 64, 100, 200):
            lines = tree.view(t)
            covered = [m for node in lines for m in range(*tree.span(node))]
            self.assertEqual(covered, list(range(t)))
            self.assertLessEqual(len(lines), tree.STATE_LEAVES + tree.STATE_COARSE)
            self.assertEqual(lines[-min(t, 16):], [(0, i) for i in range(max(0, t - 16), t)])

    def test_old_lines_are_coarser_and_stable(self):
        a, b = tree.view(150), tree.view(151)
        self.assertGreater(a[0][0], a[-1][0])
        self.assertEqual(a[:4], b[:4])

    def test_names(self):
        self.assertEqual(tree.name((3, 5)), "40+8")
        self.assertEqual(tree.name((0, 9)), "9+1")

    def test_needed_puts_dependencies_first(self):
        order = tree.needed([tree.view(60)])
        seen = set()
        for node in order:
            for dep in tree.children(node) + tree.context_view(node):
                self.assertIn(dep, seen)
            seen.add(node)


class ByteLimitTest(unittest.TestCase):
    def test_short_sources_cost_no_call(self):
        self.assertEqual(tree.no_call_text((0, 0), "user: hi"), "user: hi")
        self.assertIsNone(tree.no_call_text((0, 0), "x" * 513))
        self.assertEqual(tree.no_call_text((1, 0), ("a", "b")), "a\nb")
        self.assertIsNone(tree.no_call_text((1, 0), ("a" * 300, "b" * 300)))

    def test_too_long_lines_are_retried_at_most_five_times_and_the_shortest_kept(self):
        replies = ["x" * 900, "y" * 700, "z" * 600, "w" * 650, "v" * 800, "u" * 990, "never asked"]
        asked = []

        async def complete(prompt):
            asked.append(prompt)
            return replies[len(asked) - 1], {"input_tokens": 10, "output_tokens": 10}

        line, calls, _ = asyncio.run(build_tree.summarize("<input>\nP\n</input>", complete))
        self.assertEqual(calls, 1 + tree.RETRIES)
        self.assertEqual(line, "z" * 600)
        self.assertIn("Too long: your line is 900 bytes", asked[1])
        self.assertIn("x" * 512 + tree.RULER_NOTE, asked[1])
        self.assertTrue(asked[1].startswith("<input>\nP\n</input>"))

    def test_a_line_within_the_limit_stops_the_retries(self):
        replies = ["x" * 600, "ok line"]

        async def complete(prompt):
            return replies.pop(0), {}

        line, calls, _ = asyncio.run(build_tree.summarize("p", complete))
        self.assertEqual((line, calls), ("ok line", 2))
        self.assertLessEqual(utf8_len(line), tree.LIMIT)


class BuilderTest(unittest.TestCase):
    def test_nodes_are_cached_and_reused(self):
        msgs = [{"kind": "agent" if i % 2 else "user", "text": ("m%d " % i) * 200} for i in range(6)]
        calls = []

        async def complete(prompt):
            calls.append(prompt)
            return f"task: t{len(calls)} [in-progress]", {}

        with helpers.tmpdir() as tmp:
            cache = build_tree.NodeCache(helpers.Path(tmp) / "nodes.jsonl")
            for expected_new in (True, False):
                async def go():
                    b = build_tree.Builder(cache, 4, complete)
                    return [await b.text("s", msgs, n) for n in tree.view(6)], b.calls

                n_before = len(calls)
                texts, made = asyncio.run(go())
                self.assertEqual(len(texts), 6)
                self.assertEqual(made > 0, expected_new)
                if not expected_new:
                    self.assertEqual(len(calls), n_before)


if __name__ == "__main__":
    unittest.main()
