import json
import unittest

from helpers import FAKE_ANTHROPIC, FAKE_GITHUB, fixture_session, tmpdir

import extract
from scrub import scrub
from text import messages


class ClaudeCodeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir()
        path = fixture_session("claude_code_session.jsonl", self.tmp.name)
        self.rec = extract.parse_claude_code(path, "repo/x", None)

    def tearDown(self):
        self.tmp.cleanup()

    def prompts(self):
        return [self.rec["rows"][p["row"]]["text"] for p in self.rec["prompts"]]

    def test_only_human_prompts_are_prompts(self):
        prompts = self.prompts()
        self.assertEqual(len(prompts), 3)
        self.assertIn("Add a --verbose flag", prompts[0])
        self.assertEqual(prompts[1], "yes, and add a test for it")
        self.assertTrue(prompts[2].startswith("now write the README"))

    def test_dropped_records(self):
        texts = json.dumps(self.rec["rows"])
        self.assertNotIn("SUBAGENT TEXT", texts)  # sidechain
        self.assertNotIn("This session is being continued", texts)  # compaction summary
        self.assertNotIn("Caveat", texts)  # meta
        self.assertNotIn("def main()", texts)  # tool result content
        msgs = messages(self.rec["rows"])
        self.assertEqual([m["kind"] for m in msgs], ["user", "agent", "user", "agent", "user", "agent"])
        joined = "\n".join(m["text"] for m in msgs)
        for gone in ("task-notification", "command-name", "Request interrupted", "ide_opened_file"):
            self.assertNotIn(gone, joined)

    def test_clear_and_compact_mark_the_next_prompt(self):
        self.assertEqual([p["preceded_by"] for p in self.rec["prompts"]], ["clear", None, "compact"])

    def test_interruption_and_reply_flags(self):
        self.assertEqual([p["interrupted"] for p in self.rec["prompts"]], [False, True, False])
        self.assertTrue(all(p["replied"] for p in self.rec["prompts"]))

    def test_tool_uses_are_kept_as_rendered_lines(self):
        agent = messages(self.rec["rows"])[1]["text"]
        self.assertIn('tool: Read {"file_path":"src/cli.py"}', agent)
        self.assertIn("agent: I'll read the CLI first.", agent)

    def test_secrets_are_scrubbed_before_writing(self):
        texts = json.dumps(self.rec)
        self.assertNotIn(FAKE_GITHUB, texts)
        self.assertNotIn(FAKE_ANTHROPIC, texts)
        self.assertIn("[SECRET]", texts)

    def test_hindsight_is_kept_apart(self):
        h = self.rec["prompts"][0]["hindsight"]
        self.assertEqual(h["files_read"], 1)
        self.assertEqual(h["files_edited"], 1)
        self.assertEqual(self.rec["prompts"][2]["hindsight"]["idents_reused"] >= 0, True)


class OtherAgentsTest(unittest.TestCase):
    def test_pi_session(self):
        with tmpdir() as tmp:
            rec = extract.parse_pi(fixture_session("pi_session.jsonl", tmp), "pi/x", None)
        msgs = messages(rec["rows"])
        self.assertEqual([m["kind"] for m in msgs], ["user", "agent"])
        self.assertIn('tool: bash {"command":"pytest"} (error)', msgs[1]["text"])
        self.assertEqual(rec["project"], "/repo/pi")

    def test_codex_session(self):
        with tmpdir() as tmp:
            path = fixture_session("codex_session.jsonl", tmp)
            self.assertIs(extract.detect(path), extract.parse_codex)
            rec = extract.parse_codex(path, "codex/x", None)
        msgs = messages(rec["rows"])
        self.assertEqual([m["kind"] for m in msgs], ["user", "agent"])
        self.assertEqual(msgs[0]["text"], "Explain the recovery mode")
        self.assertIn("tool: shell", msgs[1]["text"])
        self.assertIn("agent: Recovery mode resets the bot.", msgs[1]["text"])


class ScrubTest(unittest.TestCase):
    def test_secret_patterns(self):
        samples = [FAKE_ANTHROPIC, FAKE_GITHUB, "hf_" + "a" * 34, "AKIA" + "ABCDEFGHIJKLMNOP",
                   "ya29." + "a" * 30, "glpat-" + "a" * 20, "AIza" + "a" * 35,
                   "eyJ" + "a" * 20 + ".eyJ" + "b" * 20 + "." + "c" * 20]
        for s in samples:
            self.assertNotIn(s, scrub(f"value {s} end"), s)
        self.assertEqual(scrub("Bearer " + "x" * 30), "Bearer [SECRET]")
        self.assertEqual(scrub("Api-Key " + "y" * 30), "Api-Key [SECRET]")
        self.assertEqual(scrub('password="hunter2hunter2"'), 'password="[SECRET]"')
        self.assertIn("[SECRET]@", scrub("postgres://user:pa55word@db:5432/x"))
        key = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
        self.assertEqual(scrub(key), "[SECRET]")

    def test_ordinary_text_is_untouched(self):
        text = "Rename the token parser in src/tokens.py and run pnpm test"
        self.assertEqual(scrub(text), text)


if __name__ == "__main__":
    unittest.main()
