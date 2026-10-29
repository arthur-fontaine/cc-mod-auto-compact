# Auto Compact

A Claude Code mod that notices when a prompt starts a new task and offers to compact or clear
the context before it is sent. A decision model reads a summary of the session and the new
prompt, and puts the prompt in one of five categories: it continues the current task, follows
up on it, starts a related task, starts an unrelated one, or returns to an earlier one. A
follow-up keeps everything; a related task is worth a compaction that keeps the general
picture; an unrelated one is worth a `/clear`.

The local model is **[Spisev](https://huggingface.co/arthur-fontaine/spisev-alpha-GGUF)** (spisev-alpha), a small model distilled from Jev, with
Baseten's Mammoth on spliced sessions, that runs on your machine with llama.cpp; any System One endpoint (Mammoth,
Jev) works too.

## Install

At the prompt of a terminal session:

```
/plugin install auto-compact --marketplace arthur-fontaine/cc-mod-auto-compact
```

From a shell: `claude plugin install auto-compact --marketplace arthur-fontaine/cc-mod-auto-compact`.

Then pick a provider with `/auto-compact setup`. Until then the mod does nothing.

## Providers

- **[Spisev](https://huggingface.co/arthur-fontaine/spisev-alpha-GGUF) (local)**: `/auto-compact setup` asks before downloading the GGUF (about
  1.9 GB) from Hugging Face and starts `llama-server` on 127.0.0.1 with `-hf`, for the rest of
  the session. It needs llama.cpp build 11361 or later (`llama-server` or `llama` on your PATH, or
  `AUTO_COMPACT_LLAMA_SERVER`). A `.gguf` path in `AUTO_COMPACT_MODEL` is served with `-m`.
- **Mammoth (Baseten)** or **Jev (OpenCode Zen)**: setup saves the endpoint and model; the
  key is read from `AUTO_COMPACT_API_KEY` in the environment and never stored. Any other
  System One endpoint works with `AUTO_COMPACT_ENDPOINT` and `AUTO_COMPACT_MODEL`.

## Privacy

With Spisev nothing leaves the machine. With a cloud provider, only the clipped state goes
out: the session's summary lines and the new prompt. The summaries themselves are written by
Claude Haiku through your own Claude Code session (`$.model.complete`), like any other request
of the session.

## Commands

- `/auto-compact`: the provider, the mode, the state of the summary tree and the last decision.
- `/auto-compact setup`: choose a provider.
- `/auto-compact mode off|suggest|auto`: saved across sessions.
- `/auto-compact debug`: a pane with the last prompt looked at, the summary lines sent with it (each marked `raw`, `joined` or `summary`, with its size in bytes and, when not summarized, the room left before the 512-byte limit; a summary has what it was made from dimmed under it), and the decision (each category's probability, the action and what came of it).

In **suggest** mode (the default), a prompt that looks like a related, returning or unrelated
task opens a question before it is sent: compact (or clear) first, or send it as is. Nothing
happens without your press. It asks only at a confidence of `AUTO_COMPACT_SUGGEST_CONFIDENCE`
(0.6) or more: below that, Spisev's suggestions were right half the time on the
[benchmark](#benchmark) (16 of 35), and from there on 63 of 67. In **auto** mode the mod acts only when the decision's confidence
is at least `AUTO_COMPACT_MIN_CONFIDENCE` (0.85). `continuation` and `follow_up` never act. The
mod never acts on subagents, on a timeout or an error, without a provider, or while the
previous turn's summaries are still being built (the status line says
`auto-compact: summaries catching up`).

Acting holds the prompt back (Claude Code does not compact under a prompt being submitted),
compacts with the current summary lines as what to keep, or runs `/clear`, then sends the same
prompt. A prompt with attachments is never held back. The footer shows each decision:
`context kept · 92%`, `compact? related · 71%`, `cleared · independent · 91%`; the debug log has
`[auto-compact] <category> · <confidence> → <action>`.

## Environment variables

| Variable | Default | |
| :- | :- | :- |
| `AUTO_COMPACT_PROVIDER` | none | `spisev`, or `systemone` (implied by an endpoint) |
| `AUTO_COMPACT_ENDPOINT` | Spisev: `http://127.0.0.1:8766/v1/systemone` | a System One endpoint |
| `AUTO_COMPACT_MODEL` | Spisev: `arthur-fontaine/spisev-alpha-GGUF:Q8_0` | the endpoint's model, or a GGUF for llama-server |
| `AUTO_COMPACT_AUTH` | `Bearer` | `Api-Key` for Baseten |
| `AUTO_COMPACT_API_KEY` | | the cloud provider's key, never stored |
| `AUTO_COMPACT_MODE` | `suggest` | overrides the saved mode |
| `AUTO_COMPACT_MIN_CONFIDENCE` | `0.85` | auto mode's threshold |
| `AUTO_COMPACT_SUGGEST_CONFIDENCE` | `0.6` | suggest mode's threshold |
| `AUTO_COMPACT_TIMEOUT_MS` | `4000` | a slower answer keeps the context |
| `AUTO_COMPACT_SUMMARIZER_MODEL` | `claude-haiku-5-5` | a full model id |
| `AUTO_COMPACT_DATA_DIR` | `~/.claude/plugins/data/auto-compact` | where each session's summary tree is saved |
| `AUTO_COMPACT_LLAMA_SERVER` | `llama-server`, then `llama` | |

## How it decides

After each turn the mod summarizes the new messages with Haiku into an
[OptChat](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449) tree: each message
becomes a line of at most 512 bytes, and adjacent lines merge in pairs. On each prompt it sends
the view (the last 16 messages one line each, older ones as coarser lines) and the prompt to
the decision model, clipped to 12,000 tokens, and maps the category it picks through an
editable table. The same code builds the training data
([training/README.md](training/README.md)); parity tests check that the mod and the pipeline
clip, summarize and ask alike.

## Benchmark

![Right action against median latency, one point per model. Spisev 1.7B, Elephant and Mammoth are in the fast and accurate zone; Jev sits just past its latency edge and Kev-4B well outside it; the previous Spisev and Qwen3.5-2B, trained on the v1 request, are drawn faded.](docs/benchmark.svg)

**The test**: 418 decisions, from 334 held-out prompts of 94 real coding sessions plus 84
held-out splices (the end of one session followed by the first prompt of another, where the
context should go). None of these sessions was used for training or to pick a checkpoint. The
answer key is the action two Claude teachers, Sonnet 5.5 and Opus 5.5, agreed on after seeing
what the classifier sees plus hindsight: the last 8 messages, what the agent did next
and the developer's following prompts. They agreed on 96% of the points; the rest are left
out. Always keeping the context gets 76.8% right.

**The setup**: each model got the v2 request, one at a time: the session's memory (short
lines that track its tasks and how each stands), the agent's last reply as it was written,
and the new prompt. The local models ran in llama.cpp b11669 on an Apple M4 Pro with 24 GB.
Elephant and Mammoth ran on Baseten and Jev on OpenCode Zen; their latency includes the round
trip from that Mac. The previous Spisev (alpha.1) and Qwen3.5-2B were trained on the v1
request (one-line summaries of the whole session) and got that one; they are drawn faded.

**The green zone** is where a decision model should be. Both edges are set by a rule, not
read off the results:

- **Fast**: under 400 ms at the median, the
  [Doherty threshold](https://lawsofux.com/doherty-threshold/), below which people stay
  engaged instead of waiting. The mod asks before every prompt, so this is time added to
  each one.
- **Accurate**: at least 80.4%, the lowest score that beats always keeping the context by
  more than chance on these points (one-sided exact binomial test, p < 0.05), computed by
  [`benchmark.py`](training/pipeline/benchmark.py).

| Model | Runs | Right action | Kept when it should | Acted when it should | Right action, auto mode | Right category | Latency p50 / p95 |
| :- | :- | -: | -: | -: | -: | -: | -: |
| **[Spisev 1.7B](https://huggingface.co/arthur-fontaine/spisev-alpha-GGUF)** | Local · llama.cpp · Q8_0, 1.8 GB | 92.3% | 95.6% | 81.4% | 87.8% | 83.1% | 329 / 699 ms |
| [Kev-4B](https://huggingface.co/ggml-org/Kev-4B-GGUF), zero-shot | Local · llama.cpp · Q8_0, 4.5 GB | 86.8% | 91.0% | 73.2% | 76.8% | 72.8% | 1822 / 3225 ms |
| Elephant 9B | Cloud · Baseten | 93.3% | 96.9% | 81.4% | 88.0% | 70.7% | 193 / 246 ms |
| Mammoth 27B | Cloud · Baseten | 94.7% | 96.3% | 89.7% | 90.0% | 84.4% | 233 / 340 ms |
| Jev 1.13 | Cloud · OpenCode Zen | 94.0% | 96.9% | 84.5% | 88.5% | 85.8% | 471 / 656 ms |
| [Spisev 1.7B](https://huggingface.co/arthur-fontaine/spisev-alpha-GGUF/tree/alpha.1), *alpha.1, previous version, v1 request* | Local · llama.cpp · Q8_0, 1.8 GB | 93.3% | 97.5% | 79.4% | 87.1% | 81.5% | 138 / 2590 ms |
| Qwen3.5-2B, same recipe, *previous candidate, v1 request* | Local · llama.cpp · Q8_0, 2.0 GB | 93.8% | 97.2% | 82.5% | 87.8% | 83.6% | 431 / 1913 ms |
| Always keep | | 76.8% | 100% | 0% | 76.8% | | |

- **Right action**: the model's category, through the [policy](#how-it-decides), leads to
  the teachers' action: keep, compact or clear.
- **Kept when it should**: of the 321 points where the teachers keep the context, how many
  the model keeps. A miss here is the costly one: it drops context the agent needed.
- **Acted when it should**: of the 97 points where the teachers compact or clear, how many
  the model handles the same way.
- **Right action, auto mode**: what the mod ends up doing in auto mode. It acts only at a
  confidence of 0.85 or more and within its 4 s timeout, and otherwise keeps the context.
- **Right category**: the exact category, on the 379 points where the teachers also agreed
  on it.

The table times cold requests. Since 0.1.3 the mod sends Spisev the session once a turn
ends, while the developer reads and types, so llama-server has all but the new message in
its cache; it also clips that message to 512 tokens. Measured that way on the same points,
Spisev answers in 44 / 285 ms (p50 / p95, against 329 / 699 ms cold), with no timeout.

What stands out:

- **Every model beats always keeping**, compared point by point (one-sided exact McNemar
  test, p < 0.0001 for each).
- **Spisev is level with the cloud models**: against Jev, its teacher, 10 points only
  Spisev gets right and 17 the other way (p = 0.25); against Elephant 11 and 15 (p = 0.56).
  Mammoth is ahead on most columns, 8 against 18 on right action (p = 0.076), at 16 times
  the size. Spisev needs no network or key.
- **Against the previous Spisev**, it gets the exact category more often (83.1% against
  81.5%) and acts more often when it should (81.4% against 79.4%); on right action the two
  are tied (10 points against 14, p = 0.54). It is still behind on the splices, where an
  unrelated prompt should clear the context (83.3% against 88.1%). Jev was too lenient
  there, so the training splices were relabeled by Mammoth after a first look at these
  results: this row reads slightly optimistic.
- **The long tail is gone**: v2 states are short (960 tokens at the median and 2,185 at
  most, against 1,065 and 8,179 for v1), so the p95 falls from 2.6 s to 0.7 s and no prompt
  comes near the 4 s timeout, which the previous Spisev passed on 3 of 345.
- **Kev drops context it should keep**: it compacts or clears 9.0% of the prompts the
  teachers keep, twice as often as Spisev, and is almost never confident enough for auto
  mode (one answer of 435 at 0.85 or more), so in auto mode it scores what always keeping
  does. It answers zero-shot; it was
  trained for [auto-effort](https://github.com/arthur-fontaine/cc-mod-auto-effort)'s
  question, not this one.
- **A second key agrees**: Sonnet and Opus labeled the same points again seeing only the
  raw messages, no summary line (416 points). There Spisev gets 92.8% right, the previous
  Spisev 95.0% (p = 0.11), Jev 94.7% and Mammoth 94.2% (`benchmark.py --neutral`).

To rerun it, run `uv run python pipeline/benchmark.py` then `uv run python pipeline/plot.py`
in [`training/`](training/README.md#benchmark), which also explains how the answer key and
Spisev were built.

## Development

```sh
pnpm test        # claude plugin test
pnpm validate    # claude plugin validate . --strict
pnpm eval        # the configured provider on sample decisions (reads .env)
claude --plugin-dir .
```
