import os
import subprocess
import unittest
from pathlib import Path

from helpers import REPO, CleanEnv, tmpdir

import config


class EnvTest(unittest.TestCase):
    def test_loads_a_dotenv_file_without_overriding_the_environment(self):
        with tmpdir() as tmp, CleanEnv(DECISION_MODEL="from-env"):
            path = Path(tmp) / ".env"
            path.write_text("BASETEN_API_KEY=k-file\nDECISION_MODEL=from-file\n# comment\n")
            config.load_env(path)
            self.assertEqual(os.environ["BASETEN_API_KEY"], "k-file")
            self.assertEqual(os.environ["DECISION_MODEL"], "from-env")

    def test_a_missing_file_is_not_an_error(self):
        with tmpdir() as tmp, CleanEnv():
            config.load_env(Path(tmp) / "absent.env")

    def test_missing_variables_are_named_and_exit_non_zero(self):
        with CleanEnv(DECISION_MODEL="m"):
            with self.assertRaises(SystemExit) as ctx:
                config.require("BASETEN_API_KEY", "DECISION_MODEL", "MAX_SPEND_USD")
            self.assertIn("BASETEN_API_KEY", str(ctx.exception))
            self.assertIn("MAX_SPEND_USD", str(ctx.exception))
            self.assertNotIn("DECISION_MODEL", str(ctx.exception).split(":")[1].split("(")[0])
            self.assertNotEqual(ctx.exception.code, 0)

    def test_a_script_exits_non_zero_naming_the_missing_variable(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("BASETEN", "DECISION", "MAX_SPEND"))}
        code = ("import sys; sys.path.insert(0, 'pipeline'); import config; "
                "config.load_env(config.Path('/nonexistent/.env')); config.require('BASETEN_API_KEY')")
        done = subprocess.run([os.sys.executable, "-c", code], cwd=REPO / "training", env=env,
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("BASETEN_API_KEY", done.stderr)

    def test_spend_guard_aborts_above_the_cap_and_needs_one(self):
        with CleanEnv(MAX_SPEND_USD="0"):
            self.assertTrue(config.guard_spend("t", calls=3, billed_usd=0.0, list_usd=0.1))
            with self.assertRaises(SystemExit):
                config.guard_spend("t", calls=3, billed_usd=0.01, list_usd=0.1)
        with CleanEnv():
            self.assertFalse(config.guard_spend("t", calls=3, billed_usd=1, list_usd=1, dry_run=True))
            with self.assertRaises(SystemExit):
                config.guard_spend("t", calls=3, billed_usd=0, list_usd=0)


class RedactionTest(unittest.TestCase):
    def test_known_keys_and_key_shaped_values_are_redacted(self):
        key = "bt_" + "x" * 30
        with CleanEnv(BASETEN_API_KEY=key):
            body = f'{{"error": "invalid key {key}", "detail": "Authorization: Api-Key {key}"}}'
            out = config.redact(body)
            self.assertNotIn(key, out)
            self.assertIn("[REDACTED]", out)
            self.assertNotIn("abcdefghijklmnopqrstuv", config.redact("token=abcdefghijklmnopqrstuv"))

    def test_auth_headers_are_dropped(self):
        out = config.redact_headers({"Authorization": "Api-Key secret-value-123456", "x-ratelimit-limit": "500"})
        self.assertEqual(out["Authorization"], "[REDACTED]")
        self.assertEqual(out["x-ratelimit-limit"], "500")


class GitignoreTest(unittest.TestCase):
    def test_env_is_gitignored_and_example_is_not(self):
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO).returncode
        self.assertEqual(ignored, 0, ".env must be gitignored")
        example = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=REPO).returncode
        self.assertEqual(example, 1, ".env.example must be tracked")
        for path in ("training/data/x.jsonl", "training/runs/x", "model.gguf"):
            self.assertEqual(subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode, 0, path)


if __name__ == "__main__":
    unittest.main()
