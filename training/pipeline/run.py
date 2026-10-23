"""Run the data pipeline end to end: extract, clean, points, tree, Mammoth, report.

    uv run python pipeline/run.py --sources public --limit-sessions 20     # smoke run
    uv run python pipeline/run.py --sources public                         # full run
    uv run python pipeline/run.py --dry-run                                # estimates only

Each step is its own script and can be rerun alone. Every model call is cached, so a
second identical run makes no new Haiku or Mammoth call.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(name, *args):
    print(f"\n== {name}", flush=True)
    done = subprocess.run([sys.executable, str(HERE / f"{name}.py"), *args])
    if done.returncode:
        sys.exit(done.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", choices=["public", "local", "both"], default="public")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    limit = ["--limit-sessions", str(args.limit_sessions)] if args.limit_sessions else []
    dry = ["--dry-run"] if args.dry_run else []
    step("extract", "--sources", args.sources, *limit)
    step("clean")
    step("build")
    step("build_tree", *dry)
    step("teacher", *dry)
    if not args.dry_run:
        step("report")


if __name__ == "__main__":
    main()
