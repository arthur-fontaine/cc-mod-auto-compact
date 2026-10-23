"""Settings and secrets from the repo's .env, key redaction, and the spend guard.

Every script calls `load_env()` from its `main()`, never at import, so tests can run
without the real .env. Variables already set in the environment win over the file.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # training/
REPO = ROOT.parent
DATA = ROOT / "data"
ENV_FILE = REPO / ".env"

# Names whose values are secrets: never printed, logged, cached or written.
SECRET_NAMES = ("BASETEN_API_KEY", "SPISEV_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "HF_TOKEN")


class MissingSetting(SystemExit):
    pass


def load_env(path=ENV_FILE):
    """Read `path` into os.environ without overriding what is already set."""
    from dotenv import dotenv_values

    if Path(path).exists():
        for key, value in dotenv_values(path).items():
            if value is not None and key not in os.environ:
                os.environ[key] = value
    # Model downloads stay inside the repo (gitignored), not in ~/.cache.
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
    return os.environ


def require(*names):
    """The values of `names`, or exit non-zero naming every missing one."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise MissingSetting(f"missing setting: {', '.join(missing)} (set it in .env or the environment; "
                             f"see .env.example)")
    return [os.environ[n] for n in names]


def secrets():
    return [os.environ[n] for n in SECRET_NAMES if os.environ.get(n)]


TOKENISH = re.compile(r"(?i)(api[-_ ]?key|bearer|authorization|token)(\s*[:=]?\s*)([A-Za-z0-9._\-]{16,})")


def redact(text, extra=()):
    """`text` with every known secret value, and anything shaped like a key, replaced."""
    if text is None:
        return text
    text = str(text)
    for value in [*secrets(), *extra]:
        if value and len(value) >= 6:
            text = text.replace(value, "[REDACTED]")
    return TOKENISH.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)


def redact_headers(headers):
    return {k: ("[REDACTED]" if re.search(r"(?i)auth|key|cookie|token", k) else redact(v))
            for k, v in dict(headers or {}).items()}


def max_spend():
    (value,) = require("MAX_SPEND_USD")
    try:
        return float(value)
    except ValueError:
        raise MissingSetting(f"MAX_SPEND_USD must be a number of US dollars, not {value!r}")


def guard_spend(name, *, calls, billed_usd, list_usd, dry_run=False):
    """Print the estimate; exit when the billed part is above MAX_SPEND_USD."""
    listed = "n/a (no published price)" if list_usd is None else f"${list_usd:.2f}"
    print(f"[{name}] estimate: {calls} calls, billed ${billed_usd:.2f}, list-price equivalent {listed}",
          flush=True)
    if dry_run:
        print(f"[{name}] dry run: nothing sent", flush=True)
        return False
    cap = max_spend()
    if billed_usd > cap:
        print(f"[{name}] aborting: billed estimate ${billed_usd:.2f} is above MAX_SPEND_USD ${cap:.2f}",
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    return True
