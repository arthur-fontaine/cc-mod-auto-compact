import sys
import unittest

from helpers import CleanEnv, REPO, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import make_fixtures  # noqa: E402


class FixturesTest(unittest.TestCase):
    def test_the_mods_parity_fixtures_are_up_to_date(self):
        with CleanEnv():
            expected = make_fixtures.render_module(make_fixtures.fixtures())
        path = REPO / "tests" / "fixtures" / "parity.js"
        self.assertTrue(path.exists(), "run: uv run python scripts/make_fixtures.py")
        self.assertEqual(path.read_text(), expected, "stale: run uv run python scripts/make_fixtures.py")

    def test_fixtures_use_the_real_compaction_prompt(self):
        data = make_fixtures.fixtures()
        self.assertEqual(data["compactPrompt"], (REPO / "prompts" / "compact.md").read_text())
        self.assertEqual(data["summaryRequest"]["model"], "claude-haiku-5-5")


if __name__ == "__main__":
    unittest.main()
