# Spisev

Spisev is the task-boundary classifier behind the auto-compact mod: for each new user
message in a Claude Code session, it decides how much of the current context is still
needed. The name is Latin *spissus* (compact, dense) + Jev, like Nisev (*nisus* + Jev).

This folder is everything about the model: the data pipeline, a summary tree built with
Haiku, labels from a teacher (Baseten Mammoth), a review site for human labels, and
**spisev-alpha**, a small local model distilled from Mammoth's unreviewed labels and served
by llama.cpp. It follows Nisev (cc-mod-auto-effort, training/README.md) and reuses its code
where it fits: fetch, clean, the cloud scripts, calibrate, select_checkpoint, export_gguf.

Results: see [Results](#results).

## How it decides

The classifier reads a **state**: the session so far as one-line summaries (the ≈16 most
recent messages one line each, older ones merged into coarser lines), then the new user
message whole. It puts the message in one of five categories, which
[`pipeline/policy.py`](pipeline/policy.py) (and `hooks/policy.js` in the mod) turns into an action:

| Category | Meaning | Action |
| :- | :- | :- |
| `continuation` | Carries on the current task: an answer, an approval, a correction, the next step | keep |
| `follow_up` | Builds on the task just done: a fix, refinement, test or extension that needs its details | keep |
| `related` | A new task in the same project: needs the general picture, not the recent details | compact to the view |
| `independent` | Unrelated to the conversation | clear |
| `return` | Goes back to an earlier task of the conversation | compact, keeping that branch |

The category describes the message; what to do about it lives in an editable table, not
in the weights (Nisev's lesson).

## Setup

Everything reads `../.env` (the repo root), written by you; variables already set in the
environment win. [`.env.example`](../.env.example) lists them:

- `BASETEN_API_KEY`, `DECISION_MODEL` (and optionally `DECISION_ENDPOINT`): the teacher;
- `MAX_SPEND_USD`: every model-calling script prints an estimate and aborts above it;
- `SUMMARIZER_MODEL` (default `claude-haiku-5-5`, a full id, never the `haiku` alias),
  `SUMMARIZER_EFFORT`, `SUMMARIZER_CONCURRENCY`, `SUMMARIZER_USE_API`.

A script that needs a variable that is missing says which and exits non-zero. Keys are
never printed, logged, cached or committed, and are redacted from every error body.

## Pipeline

```sh
uv sync --all-groups
uv run python pipeline/fetch.py                  # agent-trace datasets from Hugging Face (Claude Code, Pi, Codex)
uv run python pipeline/run.py --sources public --limit-sessions 20    # smoke run: extract → report
uv run python pipeline/run.py --sources public                        # full run
uv run python scripts/teacher_status.py          # labeling progress, no API call
uv run python review/server.py                   # http://127.0.0.1:8377
uv run python pipeline/export.py                 # human labels → tev1 JSONL
uv run python pipeline/dataset.py                # data/split/{train,dev,test}.jsonl
uv run python -m unittest discover -s tests
```

`run.py` runs one script per step; each can be rerun alone, and every model call is
cached, so a second identical run makes no new Haiku or Mammoth call.

### Sources and extraction

[`fetch.py`](pipeline/fetch.py) lists every `format:agent-traces` dataset and keeps those
in Claude Code, Pi or Codex format, whatever model ran them: unlike effort, a task boundary
is the user's, not the model's. `--sources local` reads `~/.claude/projects/**/*.jsonl`
read-only (never copied outside `data/`); the runs here used public data only.

[`extract.py`](pipeline/extract.py) keeps each session's main-thread rows in the shape the
mod reads from `$.session.messages()` (`role`, `text`, `toolUses`), so the tree is built from
what serving sees. Dropped, as in Nisev: tool results (only their error flag stays),
meta records, subagent messages, local commands, compaction summaries, task
notifications, interruption markers and IDE context. `/clear` and `/compact` mark the next
prompt (`preceded_by`). Hindsight is kept apart and never shown to a model: files read and
edited in the turn and their overlap with what came before, earlier identifiers reused,
later prompts that refer back. Everything is scrubbed of secrets before it is written.

[`clean.py`](pipeline/clean.py) drops forks of the same session, flattened transcripts,
scripted benchmark sources and sessions opening with a harness template (a template kept
at most 3 times, long prompts sharing their first eight words counted as one generator).
Interrupted or reply-less turns stay in the context but are no decision point. Every
filter was checked by reading random samples (`--samples`).

### Decision points, splices and splits

[`build.py`](pipeline/build.py) picks prompts with at least two messages before them
(sessions capped at 200 messages, which bounds the tree's cost). `/clear` starts a new
session file, so the first prompt after a clear is linked to the previous session of the
same project when one ended shortly before, and both are split together. Synthetic
**splices** join the end of session X to the first prompt of session Y from the same
project (or dataset), marked `synthetic: splice`, train only. Splits are by session:
test-candidates, dev and train, at most 40 train or 12 held-out points per session; dev and
test hold real points only.

### Summary tree (Haiku)

[`tree.py`](pipeline/tree.py) is the OptChat tree
([gist](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449)): a leaf is one
message in at most 512 bytes, a merge two adjacent lines in at most 512 bytes; a source
that fits costs no call. The prompt is [`prompts/compact.md`](../prompts/compact.md) with its
512-byte ruler, versioned by its hash; a line that is too long is asked again, in one
self-contained message, at most 5 times, and the shortest is kept. The view keeps the last
16 messages as leaves and merges older pairs, most due first, down to 32 lines. Budgets are
in lines, so which nodes a point needs is known before any is built, and only those are.

[`summarizer.py`](pipeline/summarizer.py) is the only Python code that calls Haiku, through the
Claude Agent SDK with the login of whoever runs it (a Claude subscription here), isolated
from that person's setup: no filesystem setting sources, safe mode (no installed plugins or
mods), no MCP servers, no tools, no slash commands, one turn. `ANTHROPIC_API_KEY` and
`ANTHROPIC_AUTH_TOKEN` are stripped unless `SUMMARIZER_USE_API=1`, and the auth mode is
printed at start. The mod's `hooks/summarizer.js` builds the same request for
`$.model.complete`; `tests/parity.test.ts` checks both on shared fixtures. A Claude usage
limit stops the summaries, keeps the cache, notifies, and prints a `CLAUDE USAGE STOP`
block with the reset time and the resume command.

### Teacher (Mammoth)

[`teacher.py`](pipeline/teacher.py) asks Baseten Mammoth through the System One API, the
state first and the question and options identical across requests so shared-prefix caching
applies. Each answer is cached by hash(model slug, request) with its whole distribution,
latency, input tokens and served version (`x-baseten-model-version-id`); a change of version
stops the run. It starts at concurrency 2 and rises slowly; a 429 halves it and backs off
with jitter. A credit or quota failure (402; a 403 or 429 mentioning credit, billing, quota,
trial or balance; five 429s in a row) stops every call, notifies, and prints a `BASETEN
STOP` block with what was labeled and how to resume.

States are clipped by one function shared with the packer and the mod
([`task.py`](pipeline/task.py) `clip_state`, `hooks/task.js`): both ends of the message are
kept, within 12,000 tokens for state + question + options (Mammoth's context is 16,384) and
a byte budget. Tokens are estimated as the larger of bytes / 3 and a count of words, digits,
whitespace runs and symbols, an upper bound of the Qwen3.5 tokenizer (code runs near 2
bytes a token).

### Report and review

[`report.py`](pipeline/report.py) prints counts per step, spend, Mammoth's category
distribution, a confidence histogram, agreement with weak labels (prompts after `/clear`
and splices should be independent or related, short acknowledgements continuation; the
precision of `independent` is the headline) and two floors: always `continuation`, and
TF-IDF + logistic regression trained on the weak labels.

[`review/server.py`](review/server.py) serves a review site on 127.0.0.1 with no external
assets: the raw transcript around the boundary, the state as the teacher saw it, and
Mammoth's pick. Blind mode (test-candidates) hides the teacher's answer until you answer;
assisted mode (train) shows it. The queue puts the lowest confidence first, plus a random
20% of the confident points. Keys: 1–5 categories, s skip, f summary insufficient, n note.
Answers append to `data/labels/human.jsonl`; [`export.py`](pipeline/export.py) writes them
as tev1 JSONL by split. Training on human labels is the beta's job, not this alpha's.

## spisev-alpha

The student is trained on every Mammoth-labeled train point, unreviewed, splices included,
against Mammoth's whole distribution over the option letters (soft cross-entropy at the
answer position, equal to the standard cross-entropy on a one-hot target; `--hard-targets`
trains on the pick alone). Nisev's recipe: LoRA rank 16 / alpha 32, loss on the answer
position only, AdamW 1e-4, cosine decay, clipping at 1, best epoch on dev.

```sh
# Qwen3-1.7B on this Mac with mlx-lm
uv run python pipeline/train.py --base Qwen/Qwen3-1.7B --out runs/qwen3-1.7b --epochs 3
uv run python pipeline/select_checkpoint.py runs/qwen3-1.7b
uv run python pipeline/fuse.py runs/qwen3-1.7b
uv run python pipeline/calibrate.py runs/qwen3-1.7b/fused-8bit

# Qwen3.5-2B on a free Colab T4 with Unsloth
uv run python cloud/pack.py
uv run python cloud/colab_train.py --base unsloth/Qwen3.5-2B --name qwen3.5-2b
uv run --group merge python cloud/import_adapter.py runs/cloud/qwen3.5-2b --base Qwen/Qwen3.5-2B
uv run python pipeline/calibrate.py runs/cloud/qwen3.5-2b/fused-8bit

# the chosen one as a llama.cpp decision model
uv run --group merge python pipeline/export_gguf.py runs/qwen3-1.7b/fused --calibrated runs/qwen3-1.7b/fused-8bit \
    --llama-cpp external/llama.cpp --name spisev-alpha-1.7b
llama-server -m runs/qwen3-1.7b/spisev-alpha-1.7b-Q8_0.gguf --port 8766 --alias spisev-alpha -c 16384

# section 11: both candidates and Mammoth on the held-out sessions
uv run python pipeline/evaluate.py --endpoint http://127.0.0.1:8766/v1/systemone --name qwen3-1.7b
uv run python pipeline/evaluate.py --mammoth --name mammoth
uv run python pipeline/alpha_report.py
```

- Qwen3.5's `--targets` lists every linear layer of the text model, the five projections of
  the linear-attention (Gated DeltaNet) layers included (`cloud/soft_loss.py`,
  `QWEN35_TARGETS`, checked against the model's own modules in `tests/test_losses.py`); the
  trainer refuses a shorter list.
- `cloud/pack.py` tokenizes here and ships token ids through the shared clip; only the split
  files and the trainer go to Colab, never `.env`. The VM checkpoints every 50 steps, and
  rerunning the same command resumes from the last one. Never run the Colab CLI at debug
  level: its log can hold OAuth tokens.
- `cloud/import_adapter.py` applies Unsloth's MLP-scaling fix and refuses to export when the
  merged model's dev agreement with Mammoth moves more than 3 points from training.
- `export_gguf.py` writes the `openjev` decision type, the calibrated temperature and a
  `systemone` template that renders exactly the training prompt (checked against the real
  Qwen chat templates in `tests/test_gguf_template.py`; llama-server counts the same tokens
  as the local tokenizer).

## Results

Filled in by the runs below.
