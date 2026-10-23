import json
import unittest

import helpers  # noqa: F401
import task
import text
from text import MARKER, clip, estimate_tokens, utf8_len


class ClipTest(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(clip("hello", 100), "hello")
        self.assertEqual(clip("hello", 100, max_tokens=10), "hello")

    def test_byte_budget_keeps_both_ends(self):
        t = "A" * 5000 + "B" * 5000
        out = clip(t, 1000)
        self.assertLessEqual(utf8_len(out), 1000)
        self.assertTrue(out.startswith("AAA") and out.endswith("BBB"))
        self.assertIn(MARKER, out)

    def test_token_budget_binds_when_tighter(self):
        t = "x" * 10_000
        out = clip(t, 9000, max_tokens=100)
        self.assertLessEqual(estimate_tokens(out), 100)
        out = clip(t, 300, max_tokens=10_000)
        self.assertLessEqual(utf8_len(out), 300)

    def test_never_splits_a_code_point(self):
        t = "😀" * 400 + "é" * 400
        out = clip(t, 500)
        out.encode("utf-8")  # valid
        self.assertLessEqual(utf8_len(out), 500)
        self.assertTrue(out.startswith("😀") and out.endswith("é"))

    def test_estimate_is_an_upper_bound_of_bytes_over_three(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 2)
        self.assertEqual(estimate_tokens("a,b,c"), 5)


class SharedClipTest(unittest.TestCase):
    """The pipeline's states, the packer and the mod all go through text.clip."""

    def test_state_fits_the_token_budget(self):
        state = task.clip_state("1+1|" + "line " * 3000, "do " * 30_000)
        self.assertLessEqual(task.sent_tokens(state), task.MAX_TOKENS)
        body = json.dumps(task.request(state, "decision-models/baseten-mammoth-27b-latest"), ensure_ascii=False)
        self.assertLess(estimate_tokens(body), 16_384 - 2_000)
        self.assertLessEqual(utf8_len(state["latest_user_message"]), task.MAX_MESSAGE_BYTES)
        self.assertTrue(state["latest_user_message"].startswith("do ") and state["latest_user_message"].endswith("do "))

    def test_packer_and_serving_use_the_same_function(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("pack", Path(helpers.ROOT / "cloud" / "pack.py"))
        source = spec.loader.get_data(spec.origin).decode()
        self.assertIn("from task import", source)
        self.assertIn("clip_state", (helpers.ROOT / "pipeline" / "train.py").read_text())
        js = (helpers.REPO / "hooks" / "text.js").read_text()
        self.assertIn("export function clip(", js)

    def test_token_estimate_covers_the_qwen_tokenizer(self):
        try:
            from tokenizers import Tokenizer
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            self.skipTest("tokenizers not installed")
        path = try_to_load_from_cache("Qwen/Qwen3.5-2B", "tokenizer.json", cache_dir=helpers.ROOT / ".hf" / "hub")
        if not isinstance(path, str):
            self.skipTest("Qwen3.5 tokenizer not in the local cache")
        tok = Tokenizer.from_file(path)
        samples = ["Add a --verbose flag to src/cli.py and update the tests.",
                   json.dumps(task.question()), "def f(x):\n    return {'a': [1, 2, 3]}\n" * 20,
                   "Explique le mode de récupération du bot, s'il te plaît.", "日本語のテキストも数える。" * 10]
        for s in samples:
            self.assertGreaterEqual(estimate_tokens(s), len(tok.encode(s).ids), s[:30])


class MessagesTest(unittest.TestCase):
    def test_agent_rows_between_prompts_form_one_message(self):
        rows = [{"role": "user", "text": "fix it", "toolUses": []},
                {"role": "assistant", "text": "Looking", "toolUses": [{"tool": "Read", "input": {"file_path": "a.py"}}]},
                {"role": "user", "text": "", "toolUses": [], "toolResults": [{"text": "x"}]},
                {"role": "assistant", "text": "", "toolUses": [{"tool": "Edit", "input": {"file_path": "a.py"},
                                                                 "isError": True}]},
                {"role": "assistant", "text": "Done", "toolUses": []},
                {"role": "user", "text": "<command-name>/clear</command-name>", "toolUses": []},
                {"role": "user", "text": "thanks", "toolUses": []}]
        msgs = text.messages(rows)
        self.assertEqual([m["kind"] for m in msgs], ["user", "agent", "user"])
        self.assertEqual(msgs[1]["text"], 'agent: Looking\ntool: Read {"file_path":"a.py"}\n'
                                          'tool: Edit {"file_path":"a.py"} (error)\nagent: Done')
        self.assertEqual(text.render(msgs[0]), "user: fix it")


if __name__ == "__main__":
    unittest.main()
