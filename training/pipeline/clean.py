"""Deduplicate sessions, drop junk, and mark which prompts may become decision points.

    uv run python pipeline/clean.py                 # data/sessions.jsonl -> data/clean.jsonl + clean-report.json
    uv run python pipeline/clean.py --samples 20    # also print random kept and dropped prompts

Each filter was checked by reading random samples (--samples), as in Nisev:

- a session republished in several repos (most "datasets" are forks of a few releases)
  is kept once, and byte-identical sessions are kept once;
- sessions whose prompts are transcripts flattened into one "user" message are dropped;
- scripted benchmark sources are dropped, and so is a session whose first prompt is a
  harness template already seen TEMPLATE_CAP times (or, for long prompts, whose first
  eight words were seen PREFIX_CAP times): a script replaying the same opening;
- within kept sessions, an over-cap template, an interrupted turn, a turn with no
  reply, or noise stays in the context but is no decision point;
- every text is scrubbed again (extract.py scrubbed it once before writing).
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, load_env  # noqa: E402
from scrub import scrub  # noqa: E402
from text import clean_prompt  # noqa: E402

SCRIPTED_SOURCES = re.compile(r"hf-coding-tools|mentionsanalysis")
FLATTENED = re.compile(r"^…\[earlier truncated\]…|ASSISTANT \(tool call\)|\bTOOL RESULT:|^USER: ")
NOISE = re.compile(r"^\[(task_started|task_notification|task_progress|init)\]|^<task-|^\s*$")
TEMPLATE_CAP = 3
PREFIX_CAP = 6
SOURCE_CAP = 1500  # decision points per source


def template(text):
    t = text.lower()
    t = re.sub(r"https?://\S+", "<url>", t)
    t = re.sub(r"(~|/|\.\.?/)[\w.\-/<>\[\]]+", "<path>", t)
    t = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b|\b[0-9a-f]{12,}\b", "<id>", t)
    t = re.sub(r"\d+", "0", t)
    return re.sub(r"[^\w<>]+", " ", t).strip()


def digest(text):
    return hashlib.sha1(text.encode()).hexdigest()


def prompt_text(session, p):
    return clean_prompt(session["rows"][p["row"]]["text"]) or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DATA / "sessions.jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "clean.jsonl")
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()
    load_env()

    sessions = [json.loads(line) for line in args.input.open()]
    report = {"input_sessions": len(sessions), "input_prompts": sum(len(s["prompts"]) for s in sessions),
              "dropped_sessions": Counter(), "not_points": Counter()}
    rng = random.Random(0)
    rng.shuffle(sessions)  # so caps keep a random sample, not the first files on disk
    owners, contents = {}, set()
    per_template, per_prefix, per_source = Counter(), Counter(), Counter()
    kept, samples = [], {"kept": [], "dropped": []}
    for s in sessions:
        prompts = [prompt_text(s, p) for p in s["prompts"]]
        why = None
        if SCRIPTED_SOURCES.search(s["source"]):
            why = "scripted_source"
        elif any(FLATTENED.search(t) for t in prompts):
            why = "flattened_transcript"
        else:
            skey = s.get("session") or s["file"]
            owner = owners.setdefault(skey, s["source"])
            content = digest("\n".join(prompts))
            if owner != s["source"]:
                why = "fork_duplicate"
            elif content in contents:
                why = "exact_duplicate"
            else:
                contents.add(content)
                first = template(prompts[0])
                pkey = digest(" ".join(first.split()[:8])) if len(prompts[0]) > 150 else None
                if per_template[digest(first)] >= TEMPLATE_CAP or (pkey and per_prefix[pkey] >= PREFIX_CAP):
                    why = "scripted_opening"
        if why:
            report["dropped_sessions"][why] += 1
            if len(samples["dropped"]) < args.samples and rng.random() < 0.2:
                samples["dropped"].append((why, prompts[0][:200]))
            continue
        for i, (p, text) in enumerate(zip(s["prompts"], prompts)):
            tkey = digest(template(text))
            pkey = digest(" ".join(template(text).split()[:8])) if len(text) > 150 else None
            reason = None
            if NOISE.search(text) or len(text) < 2:
                reason = "noise"
            elif p["interrupted"]:
                reason = "interrupted"
            elif not p["replied"]:
                reason = "no_reply"
            elif per_template[tkey] >= TEMPLATE_CAP:
                reason = "template_cap"
            elif pkey and per_prefix[pkey] >= PREFIX_CAP:
                reason = "prefix_cap"
            elif per_source[s["source"]] >= SOURCE_CAP:
                reason = "source_cap"
            per_template[tkey] += 1
            if pkey:
                per_prefix[pkey] += 1
            p["point"] = reason is None
            if reason:
                report["not_points"][reason] += 1
            else:
                per_source[s["source"]] += 1
        for r in s["rows"]:
            r["text"] = scrub(r["text"])
        kept.append(s)
        if len(samples["kept"]) < args.samples and rng.random() < 0.2:
            samples["kept"].append(rng.choice(prompts)[:200])

    kept.sort(key=lambda s: (s["source"], s["file"]))
    with args.out.open("w") as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    report.update({
        "kept_sessions": len(kept),
        "kept_prompts": sum(len(s["prompts"]) for s in kept),
        "candidate_points": sum(p.get("point", False) for s in kept for p in s["prompts"]),
        "dropped_sessions": dict(report["dropped_sessions"]),
        "not_points": dict(report["not_points"]),
        "by_source": dict(Counter(s["source"] for s in kept).most_common()),
    })
    (args.out.parent / "clean-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "by_source"}, indent=1))
    for kind, items in samples.items():
        for item in items:
            print(f"[{kind}]", item)


if __name__ == "__main__":
    main()
