"""Secret scrubbing, applied to every text before it is written under data/ or sent anywhere.

Nisev's patterns, plus Baseten and Google OAuth tokens, JWTs, URLs with credentials and
generic `key = value` assignments.
"""
import re

SECRETS = [
    (re.compile(r"\b(sk-(ant-)?[A-Za-z0-9_\-]{20,})"), "[SECRET]"),
    (re.compile(r"\b(oc_sk_[A-Za-z0-9_\-]{10,})"), "[SECRET]"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), "[SECRET]"),
    (re.compile(r"\b(glpat-[A-Za-z0-9_\-]{20,})"), "[SECRET]"),
    (re.compile(r"\b(hf_[A-Za-z0-9]{30,})"), "[SECRET]"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})"), "[SECRET]"),
    (re.compile(r"\b(AIza[0-9A-Za-z_\-]{35})"), "[SECRET]"),
    (re.compile(r"\b(ya29\.[0-9A-Za-z_\-]{20,})"), "[SECRET]"),
    (re.compile(r"\b(xox[abpr]-[A-Za-z0-9\-]{10,})"), "[SECRET]"),
    (re.compile(r"\b(sk_(live|test)_[A-Za-z0-9]{20,})"), "[SECRET]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "[SECRET]"),
    (re.compile(r"(?i)(api-key\s+)[A-Za-z0-9._\-]{16,}"), r"\1[SECRET]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{20,}"), r"\1[SECRET]"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]{3,}@"), r"\1[SECRET]@"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[SECRET]"),
    (re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret|token|access[_-]?key)(\"?\s*[:=]\s*)['\"]?[^\s'\",]{8,}"),
     r"\1\2[SECRET]"),
]


def scrub(text):
    if not text:
        return text
    for pattern, repl in SECRETS:
        text = pattern.sub(repl, text)
    return text


def scrub_value(value):
    """Scrub every string inside a JSON-like value (tool inputs)."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [scrub_value(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_value(v) for k, v in value.items()}
    return value
