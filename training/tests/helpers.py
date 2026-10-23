import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "cloud"))

# Built here so no secret-shaped literal sits in the repository.
FAKE_ANTHROPIC = "sk-" + "ant-" + "abcdefghijklmnopqrstuvwxyz0123"
FAKE_GITHUB = "gh" + "p_" + "abcdefghijklmnopqrstuvwxyz0123456789"


def fixture_session(name, tmp):
    text = (FIXTURES / name).read_text()
    text = text.replace("{{ANTHROPIC_KEY}}", FAKE_ANTHROPIC).replace("{{GITHUB_TOKEN}}", FAKE_GITHUB)
    path = Path(tmp) / name
    path.write_text(text)
    return path


class CleanEnv:
    """Run a test with a scrubbed os.environ: no real key, no real .env read."""

    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = dict(os.environ)
        for k in list(os.environ):
            if k.startswith(("BASETEN", "DECISION", "MAX_SPEND", "SUMMARIZER", "ANTHROPIC", "SPISEV")):
                del os.environ[k]
        os.environ.update(self.values)
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self.saved)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def tmpdir():
    return tempfile.TemporaryDirectory()
