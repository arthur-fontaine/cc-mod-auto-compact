# spisev — task-boundary classifier for Claude Code auto-compaction

spisev decides, for each new user message in a Claude Code session, how much
of the current context is still needed. Name: latin *spissus* (compact,
dense) + Jev, like Nisev (*nisus* + Jev).

Scope: data pipeline → Haiku summary tree → Mammoth pre-labels → review site →
spisev-alpha, a local distilled proxy of Mammoth trained on the UNREVIEWED
Mammoth-labeled dataset → GGUF export and benchmark → the spisev Claude Code
mod, built like cc-mod-auto-effort.
Out of scope: training on human labels (beta).

Repo layout: same as github.com/arthur-fontaine/cc-mod-auto-effort. The repo
root IS the mod and its own marketplace (`.claude-plugin/`, `hooks/`, `types/`,
`scripts/`, `tests/` for the mod's `.test.ts` files, `package.json`), and
everything about the model lives in `training/` (uv + Python in
`training/pipeline/`, one script per step, Python tests in `training/tests/`,
data in `training/data/`, gitignored). There is no `mod/` folder.

Follow Nisev (cc-mod-auto-effort, training/README.md) for the training side. Read that README first. Reuse its code where it
fits (fetch, clean, cloud/*, calibrate, select_checkpoint, export_gguf) rather
than rewriting it.

## 0. Secrets and configuration
- All secrets and settings live in `.env` at the repo root, written by the
  user. Scripts under `training/` load `../.env`. `.env` is gitignored; `.env.example` lists the variables, no values.
- Every script loads `.env` (python-dotenv, or `uv run --env-file .env`).
  Environment variables already set take precedence.
- Never print, log, cache, commit or copy a key. Redact keys from every error
  body before printing it. Request caches store hashes of requests, never
  headers.
- If a required variable is missing, the script says which one and exits
  non-zero; it never prompts for it.

## 1. Sources
- `--sources public|local|both`, default public.
- public: Hugging Face `format:agent-traces`, Claude Code / Pi format (Nisev fetch.py).
- local: `~/.claude/projects/**/*.jsonl`, READ-ONLY, never copied outside data/.
- Inspect real files before writing parsers; fixtures are small scrubbed excerpts.

## 2. Extract — one record per human prompt in the main thread
- Drop (Nisev): tool results, meta records, subagent messages, local commands,
  compaction summaries, task notifications, interruption markers, IDE context.
- Keep `/clear` and `/compact` as a signal on the next prompt
  (`preceded_by: clear|compact|null`).
- Keep hindsight separately: files read/edited next turn and overlap with files
  touched before, earlier identifiers reused, later user messages referring back.
- Cap sessions at 200 messages (configurable) to bound tree cost.

## 3. Clean — every filter justified by reading random samples
- Dedupe republished sessions; drop flattened transcripts; drop scripted
  benchmarks and harness prompts (template ≤3 times, first-8-words rule);
  drop interrupted or reply-less turns.
- Scrub secrets BEFORE anything is written or sent to any model.

## 4. Synthetic splices
- End of session X + first prompt of session Y, same repo (or same dataset),
  marked `synthetic: splice`. Never in held-out splits.

## 5. Summary tree (Haiku)
- OptChat tree (gist VictorTaelin/91837951a5ce5b38f341ec1ba1df6449): leaf =
  one message ≤512 bytes, merge = two adjacent lines ≤512 bytes, byte ruler,
  "too long" retry ≤5 times, keep the shortest.
- Prompt: `prompts/compact.md`, versioned by its hash.
- `pipeline/summarizer.py` is the ONLY Python code path that calls Haiku. The
  mod (section 12) has its own JS summarizer; both read prompts/compact.md and
  share model, parameters and request construction, checked by a parity test
  on shared fixtures, so training and serving see the same summaries.
- Auth: the Claude Agent SDK (Python `claude-agent-sdk`) with the Claude Code
  login of whoever runs it: Arthur's Claude subscription for this pipeline.
  In the mod, `$.model.complete` uses the end user's own session. No API key
  needed.
- No silent switch to API billing: strip `ANTHROPIC_API_KEY` and
  `ANTHROPIC_AUTH_TOKEN` from the SDK's environment unless
  `SUMMARIZER_USE_API=1`. At start, print which auth mode is in use
  (subscription or API), never the credential.
- Isolation from the user's Claude Code setup: pass explicitly (do not rely on
  SDK defaults) no filesystem setting sources, so no CLAUDE.md, settings,
  hooks, plugins or mods (auto-effort included); no MCP servers; no allowed
  tools; one turn; system prompt = prompts/compact.md. Check the option names
  against the installed SDK version.
- Model pinned by full id (`SUMMARIZER_MODEL`, default `claude-haiku-5-5`),
  never the `haiku` alias, which follows `ANTHROPIC_DEFAULT_HAIKU_MODEL`.
- Cache nodes by hash(prompt version, model, inputs) in `data/tree-cache/`.
- Every model-calling script: `--limit-sessions`, `--dry-run`, prints an
  estimate before running and spend after (list-price equivalent and number of
  calls when on a subscription); aborts above `MAX_SPEND_USD`.
- Claude usage limit (subscription limit or rate limit returned through the
  SDK): stop Haiku calls, keep the cache, run
  `osascript -e 'display notification "Claude usage limit, summaries paused" with title "spisev" sound name "Glass"'`,
  print a block starting with `CLAUDE USAGE STOP` with the reset time if
  given and the resume command, then continue only with work that needs no
  Haiku call.

## 6. Decision task (System One / tev1 format)
- state = ≈16 most recent leaves + coarser view lines + the new user message
  whole. One clip function (keeps both ends) shared by pipeline, training
  packer and serving. It enforces a token budget (≤ 12,000 tokens for state +
  question + options) as well as bytes.
- Categories, mapped to actions by `pipeline/policy.py` (Nisev lesson):
  continuation · follow_up · related · independent · return.
  Default: continuation/follow_up → keep; related → compact to view;
  independent → clear; return → compact keeping that branch.
- Teacher: Baseten Mammoth via the System One API.
  `DECISION_ENDPOINT` (default https://inference.baseten.co/v1/systemone),
  `DECISION_MODEL` (slug from the Baseten model page; prefer a versioned slug
  over `-latest`), key `BASETEN_API_KEY` from `.env`.
- Mammoth's context is 16,384 tokens. Put the state first and keep question
  and options identical across requests so shared-prefix caching applies.
- Store per call: full probability distribution, latency, input tokens, and
  any model/version field the response returns. If the served version changes
  during a run, stop and report it.

## 6b. Teacher API limits (unknown rate limit, free trial)
- Run Mammoth labeling as early as possible and cache every answer by
  hash(model slug, request); a rerun never re-asks a cached point.
- Start at concurrency 2, raise slowly; on HTTP 429 back off exponentially
  with jitter and halve concurrency. Log every 429 with its headers
  (redacted).
- Credit or quota failure — HTTP 402, a 403 or 429 whose body mentions credit,
  billing, quota, trial or balance, or 5 consecutive 429s after backoff —
  means STOP:
  1. stop every call to the teacher at once, no retry loop;
  2. keep everything already cached;
  3. run `osascript -e 'display notification "Baseten: credit or quota problem, labeling stopped" with title "spisev" sound name "Glass"'`;
  4. print a block starting with `BASETEN STOP` giving: HTTP status, the
     error body with any key redacted, points labeled so far, points left,
     and the exact command to resume;
  5. continue only with work that needs no teacher call (tests, review site,
     training on the points already labeled if they reach the minimum), then
     end the turn.
- `scripts/teacher_status.py` prints labeled / remaining / 429 count /
  last error, without calling the API.

## 7. Splits
- By session. Cap per session: 40 train points, 12 held-out.
- dev and test-candidates: real (non-synthetic) points only.

## 8. Report (`pipeline/report.py`)
- Counts per step, spend, Mammoth category distribution, confidence histogram.
- Agreement with weak labels: `/clear`-preceded prompts and splices should be
  independent/related; short acknowledgements should be continuation.
  Precision of `independent` is the headline number.
- Floors: always-continuation; TF-IDF + logistic regression on weak labels.

## 9. Review site (`training/review/`)
- 127.0.0.1 only, no external assets.
- Per point: raw transcript around the boundary (collapsible), state as the
  teacher saw it, Mammoth's pick and probabilities.
- Blind mode (test): teacher answer hidden until answered. Assisted mode
  (train): shown.
- Queue: lowest confidence first + random 20% of high-confidence points.
- Keys: 1–5 categories, s skip, f "summary insufficient", n note.
- Append-only `data/labels/human.jsonl` (mode, timestamp, teacher answer).
- `pipeline/export.py` writes tev1 JSONL from human labels, by session split.

## 10. spisev-alpha — local distilled proxy of Mammoth
- Data: every Mammoth-labeled train point, unreviewed, synthetic splices
  included. Target ≥ 3,000 train points (scale sources until reached, within
  budget).
- Targets: Mammoth's probability distribution over option letters (soft
  cross-entropy on the answer position). With a one-hot target the loss must
  equal standard cross-entropy (tested). Fallback flag: `--hard-targets`.
- Candidates, same data and recipe:
  - Qwen3-1.7B locally with mlx-lm;
  - Qwen3.5-2B on a free Colab T4 with Unsloth via google-colab-cli, using
    Nisev's cloud/ scripts and the `cloud/colab` wrapper (Colab scope only).
- Recipe (Nisev): LoRA rank 16 / alpha 32, loss on the answer position only,
  AdamW 1e-4, cosine decay, gradient clipping at 1, 3–4 epochs, best epoch on dev.
- Qwen3.5: `--targets` lists EVERY linear layer name, linear-attention
  projections included; a run adapting only attention + MLP is invalid.
- import_adapter.py applies the Unsloth MLP-scaling fix and refuses export if
  dev accuracy moves > 3 points from training.
- pack.py tokenizes locally and ships token ids, using the shared clip
  function. `.env` is never uploaded to Colab.
- Colab: checkpoint every 50 steps; rerunning resumes from the last checkpoint.
  Never run the Colab CLI at debug level; its log can hold OAuth tokens.
- Temperature calibration on dev; report calibration error before/after.
- Export the chosen model to GGUF Q8_0 with export_gguf.py, decision type and
  systemone template, served by llama-server at /v1/systemone.

## 11. Alpha evaluation (held-out sessions, Mammoth as reference)
- Top-1 agreement with Mammoth, KL to Mammoth's distribution, calibration error.
- Precision of `independent` vs weak labels, side by side with Mammoth's.
- Action agreement through policy.py.
- Latency p50/p95 via llama-server, one request at a time, vs Mammoth.
- Exact McNemar test between the two candidates (agreement with Mammoth per point).
- One-paragraph recommendation of which candidate becomes spisev-alpha.

## 12. The mod (repo root, built like cc-mod-auto-effort)
- Same layout and conventions as cc-mod-auto-effort, at the repo root:
  `.claude-plugin/` (plugin manifest and marketplace, so users run
  `claude plugin marketplace add <owner>/spisev` then
  `claude plugin install spisev@<marketplace>`), `hooks/hooks.json` +
  `hooks/register.js`, `hooks/policy.js`, `types/`, `tests/*.test.ts` run by
  `claude plugin test`, `claude plugin validate --strict`, `pnpm test` /
  `pnpm validate` / `pnpm eval`. Read its source before writing this one.
- Before using any event or mods API method, read the types Claude Code writes
  for the installed version (`.claude-plugin/types/claude-code/index.d.ts`
  after `claude --plugin-dir .`): events and methods change between
  releases. Relevant ones: `prompt.submit`, `turn.complete`, `session.end`,
  `$.session.messages`, `$.session.compact`, `$.session.usage`,
  `$.model.complete`, `$.process.spawn`, `$.ui.*`, `$.fs`.
- Commands: `/spisev` (provider, mode, tree state, last decision),
  `/spisev setup` (provider), `/spisev mode off|suggest|auto` (saved across
  sessions).
- Providers, as in auto-effort: spisev-alpha locally through llama-server,
  started by the mod on 127.0.0.1 with `-hf` on the published GGUF, asking
  before the download, stopped with the session; or any System One endpoint
  (Mammoth, Jev) with its key read from `SPISEV_API_KEY` in the environment,
  never stored. No default provider: the mod does nothing until setup.
- Summaries: after each turn (`turn.complete`), build leaves for the new
  messages and due merges in the background with `$.model.complete` on the
  pinned Haiku model, through the user's own session. Tree saved per session
  id with `$.fs` under the plugin's data directory, not in `$.store` (4 MiB,
  shared).
- Decision: on `prompt.submit`, build the state with the same clip rules as
  Python (parity test on shared fixtures), ask the provider, map the category
  through a JS port of policy.py (a test checks both tables agree, as Nisev's
  tests check hooks/policy.js).
- Acting. False positives destroy context, so:
  - `continuation` and `follow_up` never act;
  - default mode `suggest`: show the decision with a Button and hotkey to
    compact (related, return) or clear (independent) before the prompt is
    sent; nothing happens without the press;
  - `auto` acts only at confidence ≥ `SPISEV_MIN_CONFIDENCE` (default 0.85);
  - use `$.session.compact`; if its signature accepts instructions, pass the
    current view lines as what to keep. A clear must keep the prompt being
    submitted. If the installed version can't do an action safely, fall back
    to suggest and say why in `/spisev`.
- Never act on subagent requests, on timeout or error, when no provider is set
  up, or when the previous turn's summary isn't built yet (keep the context and
  show `spisev: summaries catching up`).
- Footer, like auto-effort: `spisev keep · 92%`, `spisev compact? related · 71%`,
  `spisev cleared · independent · 91%`. Debug log line
  `[spisev] <category> · <confidence> → <action>`.
- README like auto-effort: install, providers, privacy (only the clipped state
  goes to a cloud provider; nothing leaves the machine with spisev-alpha),
  commands, environment variables, how it decides, the section 11 benchmark.

## Constraints
- Tests make no network calls (stub providers, the Agent SDK, Colab) and never
  read the real `.env`.
- Nothing written outside the repo; `~/.claude` read-only.
- No raw content reaches Haiku, Mammoth or Colab before scrubbing.
- No key or OAuth token in the repo, in logs or in the transcript.
