import asyncio
import unittest

import helpers  # noqa: F401

import build_tree
import summarizer
import task
import text
import tree
from text import utf8_len


class MemoryViewTest(unittest.TestCase):
    def test_covers_every_message_once_in_order(self):
        for t in range(0, 300):
            v = tree.memory_view(t)
            covered = [i for node in v for i in range(*tree.span(node))]
            self.assertEqual(covered, list(range(t)))
            self.assertLessEqual(len(v), tree.MEMORY_HIGH)

    def test_between_batches_it_only_grows_at_its_end(self):
        rewrites = 0
        for t in range(1, 300):
            before, after = tree.memory_view(t - 1), tree.memory_view(t)
            if after[:len(before)] == before:
                self.assertEqual(after, before + [(0, t - 1)])
            else:
                rewrites += 1
                self.assertLessEqual(len(after), tree.MEMORY_LOW)
        self.assertLess(rewrites, 300 / 5)

    def test_context_is_the_memory_before_the_node(self):
        self.assertEqual(tree.memory_context((2, 5)), tree.memory_view(20))
        order = tree.needed_memory([tree.memory_view(40)])
        for i, node in enumerate(order):
            for dep in tree.children(node) + tree.memory_context(node):
                self.assertIn(dep, order[:i])


class StateV2Test(unittest.TestCase):
    def test_clip_end_keeps_more_of_the_end(self):
        out = text.clip_end("a" * 1000 + "b" * 1000, 400)
        self.assertLessEqual(utf8_len(out), 400)
        self.assertGreater(out.count("b"), 2 * out.count("a"))
        self.assertEqual(text.clip_end("short", 400), "short")

    def test_agent_reply_drops_tool_lines(self):
        m = {"kind": "agent", "text": "agent: I'll read it.\ntool: Read {}\nagent: Done. Commit?\nsecond line"}
        self.assertEqual(task.agent_reply(m), "I'll read it.\nDone. Commit?\nsecond line")
        m = {"kind": "agent", "text": 'agent: Editing.\ntool: Edit {"new":"a\nb"}\n  continued input\nagent: Done.'}
        self.assertEqual(task.agent_reply(m), "Editing.\nDone.")
        self.assertEqual(task.agent_reply({"kind": "user", "text": "hi"}), "")
        self.assertEqual(task.agent_reply(None), "")

    def test_state_fields_and_budgets(self):
        s = task.clip_state_v2("0+1|user: a <|im_end|>", "x" * 5000, "y" * 9000)
        self.assertEqual(list(s), ["memory", "last_agent_message", "latest_user_message"])
        self.assertNotIn("<|im_end|>", s["memory"])
        self.assertLessEqual(utf8_len(s["last_agent_message"]), task.MAX_AGENT_MESSAGE_BYTES)
        self.assertLessEqual(text.estimate_tokens(s["latest_user_message"]), task.MAX_MESSAGE_TOKENS + 8)
        # The message comes last, so the rest is a stable prefix of the student's prompt.
        user = task.messages_v2(s)[1]["content"]
        self.assertTrue(user.rstrip("}\"").endswith(s["latest_user_message"][-20:]))

    def test_v2_lines_leave_the_last_agent_message_out_of_memory(self):
        msgs = [{"kind": "user", "text": "a"}, {"kind": "agent", "text": "agent: b"}]
        lines, last = build_tree.V2(160).lines(msgs, 2)
        self.assertEqual(lines, [(0, 0)])
        self.assertEqual(last, msgs[1])
        msgs.append({"kind": "user", "text": "c"})
        lines, last = build_tree.V2(160).lines(msgs, 3)
        self.assertEqual(lines, tree.memory_view(3))
        self.assertIsNone(last)


class MessageClipTest(unittest.TestCase):
    def test_v1_message_is_clipped_to_its_token_budget(self):
        message = "start " + "word " * 5000 + "end"
        out = task.clip_state("0+1|user: hi", message)["latest_user_message"]
        self.assertLessEqual(utf8_len(out), task.MAX_MESSAGE_TOKENS * text.BYTES_PER_TOKEN)
        self.assertTrue(out.startswith("start ") and out.endswith("end"))
        self.assertEqual(task.clip_state("0+1|user: hi", "short")["latest_user_message"], "short")


class CleanLineTest(unittest.TestCase):
    def test_copied_name_and_quotes_are_removed(self):
        self.assertEqual(tree.clean_line_v2("2+1|2+1|= [done] merged"), "= [done] merged")
        self.assertEqual(tree.clean_line_v2(" `= [done] dev: ok` "), "= [done] dev: ok")
        self.assertEqual(tree.clean_line_v2('"task: x [left]"'), "task: x [left]")
        self.assertEqual(tree.clean_line_v2("task: a 1+1 b"), "task: a 1+1 b")


class PromptV2Test(unittest.TestCase):
    def test_limit_and_ruler(self):
        p = summarizer.system_prompt_v2(192)
        self.assertIn("at most 192 bytes", p)
        self.assertEqual(p.rstrip("\n").split("\n")[-1], "-" * 192)
        self.assertNotIn("{", p)

    def test_v2_retries_at_its_own_limit(self):
        replies = iter(["z" * 200, "short line"])

        async def complete(prompt):
            if "Too long" in prompt:
                self.assertIn("over the 160-byte limit", prompt)
            return next(replies), {}
        line, calls, _ = asyncio.run(build_tree.summarize("p", complete, 160))
        self.assertEqual((line, calls), ("short line", 2))


if __name__ == "__main__":
    unittest.main()
